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


def centering_pursuit_teacher(env: RealWorldHierCurveEnv) -> int:
    """Pure Pursuit teacher with tight-corridor wall-centering awareness."""
    pts = env.track["centerline"]
    bx, by = float(env.data.qpos[0]), float(env.data.qpos[1])
    w, x, y, z = env.data.qpos[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
    idx = int(np.argmin(dists))
    look_idx = min(idx + 2, len(pts) - 1)
    tx, ty = pts[look_idx, 0] - bx, pts[look_idx, 1] - by

    desired_yaw = math.atan2(ty, tx)
    alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    ld = max(0.14, math.hypot(tx, ty))
    kappa = 2.0 * math.sin(alpha) / ld + 0.72 * alpha

    # Gently nudge toward corridor center when either side wall is close
    rf_l = env._read_rf("rf_left")
    rf_r = env._read_rf("rf_right")
    if min(rf_l, rf_r) < 0.11:
        kappa += 0.40 * float(np.clip((rf_l - rf_r) / 0.14, -1.0, 1.0))

    if abs(kappa) < 0.42:
        return 0
    if kappa >= 1.55:
        return 3
    if kappa > 0.0:
        return 1
    if kappa <= -1.55:
        return 4
    return 2


class ValSuite28Env(RealWorldHierCurveEnv):
    """28-scenario noisy validation suite (seeds 7000-7027, disjoint from 8000+ test suite)."""

    SCENARIOS = [
        (tt, w, tr)
        for tt, w in (
            ("random", 0.44),
            ("random", 0.36),
            ("random", 0.36),
            ("pinch", 0.44),
            ("pinch", 0.44),
            ("hairpin", 0.44),
            ("chicane", 0.44),
        )
        for tr in ("sand", "gravel", "pavement", "ice")
    ]

    def __init__(self):
        super().__init__(
            terrain="sand",
            corridor_width=0.44,
            track_type="random",
            base_seed=7000,
            noise_enabled=True,
        )
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


def eval_val_suite(model: RecurrentPPO):
    env = ValSuite28Env()
    model.policy.eval()
    wins, zero_hits = 0, 0
    for _ in range(len(ValSuite28Env.SCENARIOS)):
        obs, _ = env.reset()
        st, ep, done = None, np.ones((1,), dtype=bool), False
        while not done:
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, info = env.step(int(a))
            done = tm or tc
        wins += int(info["is_success"])
        zero_hits += int(info["is_success"] and info["wall_hits"] == 0)
    return wins, zero_hits, len(ValSuite28Env.SCENARIOS)


def compute_returns(rewards, gamma=0.99):
    rets = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        rets[i] = running
    return rets


def dagger_and_critic_warmup(model: RecurrentPPO, save_dir: str):
    env = RealWorldHierCurveEnv(
        terrain="domain_rand",
        corridor_width=(0.33, 0.46),
        track_type="train_mix",
        base_seed=5100,
        noise_enabled=True,
    )

    rng = np.random.default_rng(77)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers

    # Update actor parameters only during DAgger
    actor_params = []
    for mod in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net):
        if mod is not None:
            actor_params.extend(list(mod.parameters()))
    opt_actor = torch.optim.Adam(actor_params, lr=4e-4)

    betas = (0.40, 0.25, 0.15, 0.08, 0.03, 0.00)
    eps_per_round = 44
    epochs_per_round = 14
    lr_start, lr_end = 4.0e-4, 6.0e-5

    episodes = []
    best_score = (-1, -1)
    best_path = os.path.join(save_dir, "dagger_best.zip")

    for r, beta in enumerate(betas):
        frac = r / max(1, len(betas) - 1)
        cur_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * frac))
        for pg in opt_actor.param_groups:
            pg["lr"] = cur_lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=5100 + r * 300 + ep)
            ep_obs, ep_act, ep_rew = [], [], []
            st, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                expert_act = centering_pursuit_teacher(env)
                student_act, st = model.predict(obs, state=st, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
                step_act = expert_act if (rng.random() < beta) else int(student_act)

                ep_obs.append(obs)
                ep_act.append(expert_act)
                obs, rew, tm, tc, info = env.step(step_act)
                ep_rew.append(rew)
                done = tm or tc

            wins += int(info["is_success"])
            if len(ep_obs) > 6:
                episodes.append(
                    (
                        np.asarray(ep_obs, dtype=np.float32),
                        np.asarray(ep_act, dtype=np.int64),
                        compute_returns(ep_rew, gamma=0.99),
                    )
                )

        model.policy.train()
        order = np.arange(len(episodes))
        for _ in range(epochs_per_round):
            rng.shuffle(order)
            total_loss = 0.0
            for idx in order:
                obs_np, act_np, _ = episodes[idx]
                obs_t = torch.as_tensor(obs_np, device=device)
                act_t = torch.as_tensor(act_np, device=device)
                starts = torch.zeros(len(obs_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)

                dist, _ = model.policy.get_distribution(obs_t, (h0, c0), starts)
                loss = F.cross_entropy(dist.distribution.logits, act_t)
                opt_actor.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
                opt_actor.step()
                total_loss += float(loss.item())

        vw, vz, vn = eval_val_suite(model)
        marker = ""
        if (vw, vz) >= best_score:
            best_score = (vw, vz)
            model.save(best_path)
            marker = " <-- NEW BEST"

        print(
            f"[DAgger R{r} beta={beta:.2f} lr={cur_lr:.1e}] rollout: {wins:2d}/{eps_per_round} "
            f"| CE: {total_loss/len(episodes):.4f} | val28: {vw:2d}/{vn} (0-hit: {vz:2d}){marker}"
        )

    # Reload best DAgger actor weights before calibrating the critic
    best_model = RecurrentPPO.load(best_path, env=model.get_env(), device=device)
    best_model.learning_rate = 3.5e-5
    best_model.ent_coef = 0.001
    best_model.target_kl = 0.008

    # Stage B: Pre-calibrate critic on the collected rollouts (zero actor drift)
    critic_params = []
    for mod in (
        best_model.policy.lstm_critic,
        best_model.policy.mlp_extractor.value_net,
        best_model.policy.value_net,
    ):
        if mod is not None:
            critic_params.extend(list(mod.parameters()))

    opt_critic = torch.optim.Adam(critic_params, lr=1.5e-3)
    best_model.policy.train()
    # Use the latest 120 most on-policy episodes for critic calibration
    calib_eps = episodes[-120:]
    c_order = np.arange(len(calib_eps))
    for ep_i in range(20):
        rng.shuffle(c_order)
        mse_sum = 0.0
        for idx in c_order:
            obs_np, _, ret_np = calib_eps[idx]
            obs_t = torch.as_tensor(obs_np, device=device)
            ret_t = torch.as_tensor(ret_np, device=device)
            starts = torch.zeros(len(obs_np), dtype=torch.float32, device=device)
            starts[0] = 1.0
            h0 = torch.zeros((n_layers, 1, h_dim), device=device)
            c0 = torch.zeros((n_layers, 1, h_dim), device=device)

            vals = best_model.policy.predict_values(obs_t, (h0, c0), starts).squeeze(-1)
            v_loss = F.mse_loss(vals, ret_t)
            opt_critic.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            opt_critic.step()
            mse_sum += float(v_loss.item())
        if ep_i in (0, 9, 19):
            print(f"[Critic Calib epoch {ep_i+1:2d}/20] value MSE: {mse_sum/len(calib_eps):.2f}")

    best_model.save(best_path)
    return best_model


def make_env(rank: int):
    def _init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return Monitor(
            RealWorldHierCurveEnv(
                terrain="domain_rand",
                corridor_width=(0.33, 0.46),
                track_type="train_mix",
                base_seed=2000 + rank * 100,
                noise_enabled=True,
            )
        )

    return _init


def safe_close(vec_env):
    try:
        vec_env.close()
    except (BrokenPipeError, EOFError, OSError):
        pass


if __name__ == "__main__":
    start_ckpt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "checkpoints/hier_curve_generalist/generalist_polished.zip"
    )
    ppo_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 80_000

    save_dir = "checkpoints/hier_curve_95plus"
    log_dir = f"{save_dir}_logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    n_envs = 8
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: Monitor(ValSuite28Env())])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=8_000 // n_envs,
        n_eval_episodes=28,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=40_000 // n_envs,
        save_path=save_dir,
        name_prefix="step",
    )

    model = RecurrentPPO.load(
        start_ckpt,
        env=train_env,
        custom_objects={"learning_rate": 3.5e-5, "ent_coef": 0.001, "clip_range": 0.12},
    )
    model.target_kl = 0.008
    print(f"Loaded starting checkpoint: {start_ckpt}")

    model = dagger_and_critic_warmup(model, save_dir)
    print("\nStarting safety-rewarded KL-guarded PPO polish...")

    try:
        model.learn(
            total_timesteps=ppo_steps,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest checkpoint...")
        model.save(f"{save_dir}/interrupted_model.zip")
    else:
        model.save(f"{save_dir}/final_model.zip")
        print(f"Saved {save_dir}/final_model.zip")
    finally:
        safe_close(train_env)
        safe_close(eval_env)
