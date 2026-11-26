import os
import sys
import math
import signal
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env.hier_curve_env import RealWorldHierCurveEnv


def discrete_pursuit_teacher(env: RealWorldHierCurveEnv) -> int:
    pts = env.track["centerline"]
    bx, by = float(env.data.qpos[0]), float(env.data.qpos[1])
    w, x, y, z = env.data.qpos[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
    idx = int(np.argmin(dists))

    # Tighter 2-segment (0.28 m) lookahead avoids apex clipping in 0.33 m pinches & hairpins
    look_idx = min(idx + 2, len(pts) - 1)
    tx, ty = pts[look_idx, 0] - bx, pts[look_idx, 1] - by

    desired_yaw = math.atan2(ty, tx)
    alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    ld = max(0.14, math.hypot(tx, ty))
    kappa = 2.0 * math.sin(alpha) / ld + 0.70 * alpha

    if abs(kappa) < 0.50:
        return 0
    if kappa >= 1.60:
        return 3
    if kappa > 0.0:
        return 1
    if kappa <= -1.60:
        return 4
    return 2


class GeneralistEvalEnv(RealWorldHierCurveEnv):
    """Deterministic 20-scenario validation suite (seeds 7000-7019, disjoint from 8000+ test suite)."""

    SCENARIOS = [
        (tt, w, tr)
        for tt, w in (
            ("random", 0.44),
            ("random", 0.36),
            ("hairpin", 0.44),
            ("chicane", 0.44),
            ("pinch", 0.44),
        )
        for tr in ("sand", "gravel", "pavement", "ice")
    ]

    def __init__(self):
        super().__init__(terrain="sand", corridor_width=0.44, track_type="random", base_seed=7000, noise_enabled=False)
        self._slot = 0

    def reset(self, *, seed=None, options=None):
        slot = self._slot % len(self.SCENARIOS)
        self._slot += 1
        tt, w, tr = self.SCENARIOS[slot]
        self.track_type = tt
        self.corridor_width = w
        self.terrain_mode = tr
        self.max_steps = 280
        return super().reset(seed=7000 + slot, options=options)


def make_env(rank: int, terrain: str = "random", corridor_width=0.44, track_type: str = "random"):
    def _init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        env = RealWorldHierCurveEnv(
            terrain=terrain,
            corridor_width=corridor_width,
            track_type=track_type,
            base_seed=1000 + rank * 100,
            noise_enabled=True,
        )
        return Monitor(env)

    return _init


def transfer_hier_base(model: RecurrentPPO, hier_path: str):
    base = RecurrentPPO.load(hier_path, device="cpu")
    src_sd = base.policy.state_dict()
    dst_sd = model.policy.state_dict()

    copied = []
    for k, v in src_sd.items():
        if k in dst_sd and v.shape == dst_sd[k].shape and not k.startswith("action_net"):
            dst_sd[k] = v.to(dst_sd[k].device)
            copied.append(k)

    model.policy.load_state_dict(dst_sd)
    print(f"Warm-started {len(copied)} compatible LSTM/MLP tensors from {hier_path}")


def dagger_warmstart(
    model: RecurrentPPO,
    corridor_width=(0.34, 0.46),
    terrain: str = "domain_rand",
    track_type: str = "train_mix",
    betas=(1.0, 0.7, 0.45, 0.25, 0.10, 0.0),
    eps_per_round: int = 32,
    epochs_per_round: int = 16,
    lr_start: float = 1.2e-3,
    lr_end: float = 2.0e-4,
):
    env = RealWorldHierCurveEnv(
        terrain=terrain,
        corridor_width=corridor_width,
        track_type=track_type,
        base_seed=4200,
        noise_enabled=True,
    )
    eval_env = GeneralistEvalEnv()

    rng = np.random.default_rng(123)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr_start)
    episodes = []

    for r, beta in enumerate(betas):
        # Smooth cosine LR schedule across rounds
        frac = r / max(1, len(betas) - 1)
        cur_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=4200 + r * 300 + ep)
            ep_obs, ep_act = [], []
            state, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                expert_act = discrete_pursuit_teacher(env)
                student_act, state = model.predict(
                    obs, state=state, episode_start=ep_start, deterministic=True
                )
                ep_start = np.zeros((1,), dtype=bool)
                step_act = expert_act if (rng.random() < beta) else int(student_act)

                ep_obs.append(obs)
                ep_act.append(expert_act)
                obs, _, term, trunc, info = env.step(step_act)
                done = term or trunc

            wins += int(info["is_success"])
            if len(ep_obs) > 6:
                episodes.append(
                    (np.asarray(ep_obs, dtype=np.float32), np.asarray(ep_act, dtype=np.int64))
                )

        model.policy.train()
        order = np.arange(len(episodes))
        for _ in range(epochs_per_round):
            rng.shuffle(order)
            total_loss = 0.0
            for idx in order:
                obs_np, act_np = episodes[idx]
                obs_t = torch.as_tensor(obs_np, device=device)
                act_t = torch.as_tensor(act_np, device=device)
                starts = torch.zeros(len(obs_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)

                dist, _ = model.policy.get_distribution(obs_t, (h0, c0), starts)
                loss = F.cross_entropy(dist.distribution.logits, act_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())

        print(
            f"[DAgger R{r} beta={beta:.2f} lr={cur_lr:.1e}] rollout solve: {wins:2d}/{eps_per_round} "
            f"| dataset: {len(episodes):3d} eps | CE loss: {total_loss/len(episodes):.4f}"
        )

    model.policy.eval()
    eval_env._slot = 0
    val_wins = 0
    for _ in range(len(GeneralistEvalEnv.SCENARIOS)):
        obs, _ = eval_env.reset()
        state, ep_start, done = None, np.ones((1,), dtype=bool), False
        while not done:
            act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = eval_env.step(int(act))
            done = term or trunc
        val_wins += int(info["is_success"])
    print(f"Post-DAgger 20-scenario OOD validation solve rate (seeds 7000-7019): {val_wins}/20")


def safe_close(vec_env):
    try:
        vec_env.close()
    except (BrokenPipeError, EOFError, OSError):
        pass


if __name__ == "__main__":
    generalist = "--generalist" in sys.argv
    refresh_dagger = "--refresh" in sys.argv
    pos_args = [a for a in sys.argv[1:] if a not in ("--generalist", "--refresh")]

    arg_ckpt = pos_args[0] if len(pos_args) > 0 else None
    additional_steps = int(pos_args[1]) if len(pos_args) > 1 else 100_000
    corridor_width = (0.34, 0.46) if generalist else (float(pos_args[2]) if len(pos_args) > 2 else 0.44)

    save_dir = "checkpoints/hier_curve_generalist" if generalist else "checkpoints/hier_curve"
    log_dir = f"{save_dir}_logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_terrain = "domain_rand" if generalist else "random"
    train_track = "train_mix" if generalist else "random"

    n_envs = 8
    train_env = SubprocVecEnv(
        [make_env(i, train_terrain, corridor_width, train_track) for i in range(n_envs)]
    )
    if generalist:
        eval_env = DummyVecEnv([lambda: Monitor(GeneralistEvalEnv())])
        n_eval_eps = 20
    else:
        eval_env = DummyVecEnv(
            [lambda: Monitor(RealWorldHierCurveEnv(terrain="cycle", corridor_width=corridor_width, base_seed=500, noise_enabled=False))]
        )
        n_eval_eps = 16

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=8_000 // n_envs,
        n_eval_episodes=n_eval_eps,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=40_000 // n_envs,
        save_path=save_dir,
        name_prefix="hier_curve_step",
    )

    loaded_exact = False
    if arg_ckpt and os.path.exists(arg_ckpt):
        try:
            model = RecurrentPPO.load(
                arg_ckpt,
                env=train_env,
                custom_objects={"ent_coef": 0.002, "learning_rate": 6e-5},
            )
            model.target_kl = 0.012
            print(f"Resumed hierarchical curve checkpoint: {arg_ckpt}")
            if generalist or refresh_dagger:
                print("Running generalist 6-round DAgger warm-start across OOD geometries & physics...")
                dagger_warmstart(
                    model,
                    corridor_width=corridor_width,
                    terrain=train_terrain,
                    track_type=train_track,
                )
                model.save(f"{save_dir}/generalist_dagger.zip")
                print(f"Saved {save_dir}/generalist_dagger.zip")
            loaded_exact = True
            reset_steps = generalist
        except ValueError:
            print(f"Space mismatch on {arg_ckpt} — transferring LSTM base + running discrete DAgger...")

    if not loaded_exact:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=1.2e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.003,
            target_kl=0.012,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                net_arch=dict(pi=[64], vf=[64]),
            ),
            verbose=1,
        )
        hier_base = arg_ckpt if (arg_ckpt and os.path.exists(arg_ckpt)) else "checkpoints/maze_5x5_hier_best.zip"
        if os.path.exists(hier_base):
            transfer_hier_base(model, hier_base)
        dagger_warmstart(model, corridor_width=corridor_width, terrain=train_terrain, track_type=train_track)
        model.save(f"{save_dir}/generalist_dagger.zip")
        reset_steps = True

    try:
        model.learn(
            total_timesteps=additional_steps,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=reset_steps,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest checkpoint...")
        model.save(f"{save_dir}/interrupted_model.zip")
        print(f"Saved {save_dir}/interrupted_model.zip")
    else:
        model.save(f"{save_dir}/final_model.zip")
        print(f"Saved {save_dir}/final_model.zip")
    finally:
        safe_close(train_env)
        safe_close(eval_env)