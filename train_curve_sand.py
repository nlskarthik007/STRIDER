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

from env.realworld_curve_env import RealWorldCurveEnv


def pure_pursuit_teacher(env: RealWorldCurveEnv) -> np.ndarray:
    pts = env.track["centerline"]
    bx, by = float(env.data.qpos[0]), float(env.data.qpos[1])
    w, x, y, z = env.data.qpos[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
    idx = int(np.argmin(dists))
    look_idx = min(idx + 3, len(pts) - 1)
    tx, ty = pts[look_idx, 0] - bx, pts[look_idx, 1] - by

    desired_yaw = math.atan2(ty, tx)
    alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    ld = max(0.15, math.hypot(tx, ty))
    kappa = 2.0 * math.sin(alpha) / ld

    v_des = 0.50 / (1.0 + 0.75 * abs(kappa))
    w_des = v_des * kappa + 2.2 * alpha - 0.12 * env._read_gyro_z()

    act_v = (v_des - 0.30) / 0.25
    act_w = w_des / 3.00
    return np.clip([act_v, act_w], -1.0, 1.0).astype(np.float32)


def make_env(rank: int, terrain: str = "random", corridor_width: float = 0.44):
    def _init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        env = RealWorldCurveEnv(
            terrain=terrain,
            corridor_width=corridor_width,
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


def bc_warmstart(
    model: RecurrentPPO,
    corridor_width: float = 0.44,
    n_init: int = 24,
    dagger_rounds: int = 3,
    n_dagger: int = 16,
    epochs_per_round: int = 15,
):
    env = RealWorldCurveEnv(
        terrain="random", corridor_width=corridor_width, base_seed=4200, noise_enabled=True
    )
    eval_env = RealWorldCurveEnv(
        terrain="random", corridor_width=corridor_width, base_seed=9500, noise_enabled=False
    )

    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers
    opt = torch.optim.Adam(model.policy.parameters(), lr=1.5e-3)
    episodes = []

    for r in range(dagger_rounds + 1):
        n_eps = n_init if r == 0 else n_dagger
        use_teacher = (r == 0)
        wins = 0

        model.policy.eval()
        for ep in range(n_eps):
            obs, _ = env.reset(seed=4200 + r * 200 + ep)
            ep_obs, ep_act = [], []
            state, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                expert_act = pure_pursuit_teacher(env)
                if use_teacher:
                    step_act = expert_act
                else:
                    step_act, state = model.predict(
                        obs, state=state, episode_start=ep_start, deterministic=True
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                ep_obs.append(obs)
                ep_act.append(expert_act)  # always label with teacher recovery action
                obs, _, term, trunc, info = env.step(step_act)
                done = term or trunc
            wins += int(info["is_success"])
            if len(ep_obs) > 8:
                episodes.append(
                    (np.asarray(ep_obs, dtype=np.float32), np.asarray(ep_act, dtype=np.float32))
                )

        model.policy.train()
        for _ in range(epochs_per_round):
            total_loss = 0.0
            for obs_np, act_np in episodes:
                obs_t = torch.as_tensor(obs_np, device=device)
                act_t = torch.as_tensor(act_np, device=device)
                starts = torch.zeros(len(obs_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)

                dist, _ = model.policy.get_distribution(obs_t, (h0, c0), starts)
                loss = F.mse_loss(dist.distribution.mean, act_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())

        tag = "Init Teacher" if r == 0 else f"DAgger R{r}"
        print(
            f"[{tag}] rollout solve: {wins}/{n_eps} | dataset: {len(episodes)} eps "
            f"| MSE: {total_loss/len(episodes):.4f}"
        )

    model.policy.eval()
    student_wins = 0
    for s in range(9500, 9505):
        obs, _ = eval_env.reset(seed=s)
        state, ep_start, done = None, np.ones((1,), dtype=bool), False
        while not done:
            act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = eval_env.step(act)
            done = term or trunc
        student_wins += int(info["is_success"])
    print(f"Post-DAgger local-sensor student solve rate: {student_wins}/5")

def safe_close(vec_env):
    try:
        vec_env.close()
    except (BrokenPipeError, EOFError, OSError):
        pass


if __name__ == "__main__":
    arg_ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    additional_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250_000
    corridor_width = float(sys.argv[3]) if len(sys.argv) > 3 else 0.44

    os.makedirs("checkpoints/curve_sand", exist_ok=True)
    os.makedirs("checkpoints/curve_sand_logs", exist_ok=True)

    n_envs = 8
    train_env = SubprocVecEnv([make_env(i, "random", corridor_width) for i in range(n_envs)])
    eval_env = DummyVecEnv(
    [lambda: Monitor(RealWorldHierCurveEnv(terrain="cycle", corridor_width=corridor_width, base_seed=500, noise_enabled=False))]
)

    eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="checkpoints/hier_curve",
    log_path="checkpoints/hier_curve_logs",
    eval_freq=8_000 // n_envs,
    n_eval_episodes=16,  # exact 4 episodes per terrain (sand, gravel, pavement, ice)
    deterministic=True,
)
    ckpt_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path="checkpoints/curve_sand",
        name_prefix="curve_sand_step",
    )

    loaded_exact = False
    if arg_ckpt and os.path.exists(arg_ckpt):
        try:
            model = RecurrentPPO.load(
                arg_ckpt,
                env=train_env,
                custom_objects={"ent_coef": 0.001, "learning_rate": 2e-4},
            )
            print(f"Resumed curve checkpoint directly: {arg_ckpt}")
            loaded_exact = True
            reset_steps = False
        except ValueError:
            print(f"Space mismatch on {arg_ckpt} — building continuous policy with base transfer + BC...")

    if not loaded_exact:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.001,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                net_arch=dict(pi=[64], vf=[64]),
                log_std_init=-1.2,
            ),
            verbose=1,
        )
        hier_base = arg_ckpt if (arg_ckpt and os.path.exists(arg_ckpt)) else "checkpoints/maze_5x5_hier_best.zip"
        if os.path.exists(hier_base):
            transfer_hier_base(model, hier_base)
        bc_warmstart(model, corridor_width=corridor_width)
        model.save("checkpoints/curve_sand/bc_init.zip")
        reset_steps = True

    try:
        model.learn(
            total_timesteps=additional_steps,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=reset_steps,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest checkpoint...")
        model.save("checkpoints/curve_sand/interrupted_model.zip")
        print("Saved checkpoints/curve_sand/interrupted_model.zip")
    else:
        model.save("checkpoints/curve_sand_final.zip")
        print("Saved checkpoints/curve_sand_final.zip")
    finally:
        safe_close(train_env)
        safe_close(eval_env)