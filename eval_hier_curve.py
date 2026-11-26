import sys
import numpy as np
import mujoco.viewer
from sb3_contrib import RecurrentPPO
from env.hier_curve_env import RealWorldHierCurveEnv


def run_benchmark(ckpt: str, corridor_width: float = 0.44):
    model = RecurrentPPO.load(ckpt)
    for terrain in ("sand", "gravel", "pavement", "ice"):
        env = RealWorldHierCurveEnv(
            terrain=terrain, corridor_width=corridor_width, base_seed=500, noise_enabled=False
        )
        wins, progs, hits = 0, [], []
        for s in range(500, 510):
            obs, _ = env.reset(seed=s)
            state, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
                obs, _, term, trunc, info = env.step(int(act))
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
        ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/hier_curve/best_model.zip"
        width = float(sys.argv[3]) if len(sys.argv) > 3 else 0.44
        run_benchmark(ckpt, width)
        return

    terrain = sys.argv[1] if len(sys.argv) > 1 else "sand"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/hier_curve/best_model.zip"
    width = float(sys.argv[4]) if len(sys.argv) > 4 else 0.44

    model = RecurrentPPO.load(ckpt)
    env = RealWorldHierCurveEnv(
        terrain=terrain, corridor_width=width, base_seed=seed, noise_enabled=False, render_realtime=True
    )
    obs, info = env.reset(seed=seed)
    print(f"Track seed={info['track_seed']} | terrain={info['terrain']} | width={width:.2f}m")

    state, ep_start = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [2.2, 0.0, 0.0]
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -65.0

        while viewer.is_running():
            act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, step_info = env.step(int(act))
            if term or trunc:
                status = "GOAL REACHED" if step_info["is_success"] else "ENDED"
                print(
                    f"{status} | steps={env.steps} | progress={step_info['progress_m']:.2f}m "
                    f"| wall_hits={step_info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    main()