import sys
import numpy as np
import mujoco.viewer
from env.realworld_curve_env import RealWorldCurveEnv


def tangential_demo_driver(obs: np.ndarray) -> np.ndarray:
    rf_f, _, rf_l, rf_r, rf_fl, rf_fr = obs[:6] * 1.20
    gyro_z = obs[7] * 4.0
    steer = 2.8 * (rf_fl - rf_fr) + 1.8 * (rf_l - rf_r) - 0.12 * gyro_z
    clearance = min(rf_f, rf_fl, rf_fr)
    speed = 0.80 if clearance > 0.38 else (0.20 if clearance > 0.22 else -0.40)
    return np.clip([speed, steer], -1.0, 1.0).astype(np.float32)


def run_benchmark(ckpt: str, corridor_width: float = 0.44):
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(ckpt)
    for terrain in ("sand", "gravel", "pavement", "ice"):
        env = RealWorldCurveEnv(
            terrain=terrain, corridor_width=corridor_width, base_seed=500, noise_enabled=False
        )
        wins, progs, hits = 0, [], []
        for s in range(500, 510):
            obs, _ = env.reset(seed=s)
            state, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
            wins += int(info["is_success"])
            progs.append(info["progress_m"])
            hits.append(info["wall_hits"])
        print(
            f"{terrain:8s} | solve: {wins:2d}/10 | avg progress: {np.mean(progs):.2f}/5.74 m "
            f"| avg hits: {np.mean(hits):.1f}"
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/curve_sand/best_model.zip"
        width = float(sys.argv[3]) if len(sys.argv) > 3 else 0.44
        run_benchmark(ckpt, width)
        return

    terrain = sys.argv[1] if len(sys.argv) > 1 else "sand"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "--demo"
    width = float(sys.argv[4]) if len(sys.argv) > 4 else 0.44

    model = None
    if ckpt != "--demo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(ckpt)

    env = RealWorldCurveEnv(
        terrain=terrain,
        corridor_width=width,
        base_seed=seed,
        noise_enabled=False,
        render_realtime=True,
    )
    obs, info = env.reset(seed=seed)
    print(f"Track seed={info['track_seed']} | terrain={info['terrain']} | width={width:.2f}m")

    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [2.2, 0.0, 0.0]
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -65.0

        while viewer.is_running():
            if model is not None:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=ep_start, deterministic=True
                )
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action = tangential_demo_driver(obs)

            obs, reward, terminated, truncated, step_info = env.step(action)
            if terminated or truncated:
                status = "GOAL REACHED" if step_info["is_success"] else "ENDED"
                print(
                    f"{status} | steps={env.steps} | progress={step_info['progress_m']:.2f}m "
                    f"| wall_hits={step_info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    main()