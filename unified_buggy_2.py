import sys
import numpy as np
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.hier_curve_env import RealWorldHierCurveEnv
from env.hier_macro_maze_env import RealWorldMacroMazeEnv


CURVE_CKPT = "checkpoints/hier_curve_champion_98pct.zip"
MAZE_CKPT = "checkpoints/hier_maze_terrain_95pct.zip"


def run_unified_eval():
    curve_model = RecurrentPPO.load(CURVE_CKPT)
    maze_model = RecurrentPPO.load(MAZE_CKPT)

    print("=== UNIFIED RL_BUGGY_2.0 BENCHMARK (NOISE ON | ALL TERRAINS) ===")

    # 1. v1 Capability: Zero-shot 7x7 branching mazes with dead-end backtracking
    m_wins, m_zh, m_tot = 0, 0, 0
    for tr in ("pavement", "sand", "gravel", "ice", "mud"):
        env = RealWorldMacroMazeEnv(grid_size=7, terrain=tr, cell_size=0.44, noise_enabled=True)
        w, zh = 0, 0
        for s in range(9600, 9610):
            obs, _ = env.reset(seed=s)
            st, ep, done = None, np.ones((1,), dtype=bool), False
            while not done:
                a, st = maze_model.predict(obs, state=st, episode_start=ep, deterministic=True)
                ep = np.zeros((1,), dtype=bool)
                obs, _, tm, tc, info = env.step(int(a))
                done = tm or tc
            w += int(info["is_success"])
            zh += int(info["is_success"] and info["wall_hits"] == 0)
        m_wins += w
        m_zh += zh
        m_tot += 10
        print(f"[v1 Maze 7x7]   {tr:9s} | solve: {w:2d}/10 (0-hit: {zh:2d})")

    # 2. v2 Capability: Continuous curves, 180° hairpins, chicanes, pinches
    c_wins, c_zh, c_tot = 0, 0, 0
    for ttype, width, tr in (
        ("random", 0.44, "mud"),
        ("random", 0.36, "ice"),
        ("hairpin", 0.44, "ice"),
        ("chicane", 0.44, "sand"),
        ("pinch", 0.44, "ice"),
    ):
        env = RealWorldHierCurveEnv(terrain=tr, corridor_width=width, track_type=ttype, noise_enabled=True)
        w, zh = 0, 0
        for s in range(8500, 8506):
            obs, _ = env.reset(seed=s)
            st, ep, done = None, np.ones((1,), dtype=bool), False
            while not done:
                a, st = curve_model.predict(obs, state=st, episode_start=ep, deterministic=True)
                ep = np.zeros((1,), dtype=bool)
                obs, _, tm, tc, info = env.step(int(a))
                done = tm or tc
            w += int(info["is_success"])
            zh += int(info["is_success"] and info["wall_hits"] == 0)
        c_wins += w
        c_zh += zh
        c_tot += 6
        print(f"[v2 {ttype:9s}] {tr:9s} | solve: {w:2d}/6  (0-hit: {zh:2d})")

    print(
        f"\nTOTAL UNIFIED SCORE: {m_wins + c_wins}/{m_tot + c_tot} "
        f"({100.0 * (m_wins + c_wins) / (m_tot + c_tot):.1f}%) | "
        f"0-hit: {m_zh + c_zh}/{m_tot + c_tot}"
    )


def run_live(mode: str = "maze", terrain: str = "ice", seed: int = 9600, grid_size: int = 7):
    if mode == "maze":
        model = RecurrentPPO.load(MAZE_CKPT)
        env = RealWorldMacroMazeEnv(grid_size=grid_size, terrain=terrain, cell_size=0.44, noise_enabled=True, render_realtime=True)
        obs, _ = env.reset(seed=seed)
        c_mid = 0.5 * (grid_size - 1) * 0.44
        lookat, dist = [c_mid, c_mid, 0.0], 4.4 if grid_size <= 7 else 5.6
    else:
        model = RecurrentPPO.load(CURVE_CKPT)
        env = RealWorldHierCurveEnv(terrain=terrain, corridor_width=0.44, track_type=mode, noise_enabled=True, render_realtime=True)
        obs, _ = env.reset(seed=seed)
        pts = env.track["centerline"]
        lookat = [float(pts[:, 0].mean()), float(pts[:, 1].mean()), 0.0]
        dist = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 2.5) * 1.35

    st, ep = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = dist
        viewer.cam.elevation = -72.0
        while viewer.is_running():
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, info = env.step(int(a))
            if tm or tc:
                print(f"{'GOAL REACHED' if info['is_success'] else 'ENDED'} | hits={info['wall_hits']}")
                break


if __name__ == "__main__":
    if "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        mode = args[0] if len(args) > 0 else "maze"       # maze | hairpin | chicane | pinch | random
        terrain = args[1] if len(args) > 1 else "ice"
        seed = int(args[2]) if len(args) > 2 else 9600
        grid = int(args[3]) if len(args) > 3 else 7
        run_live(mode=mode, terrain=terrain, seed=seed, grid_size=grid)
    else:
        run_unified_eval()
