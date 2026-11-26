import sys
import numpy as np
import mujoco.viewer
from sb3_contrib import RecurrentPPO
from env.hier_curve_env import RealWorldHierCurveEnv


def eval_block(
    model: RecurrentPPO,
    label: str,
    terrain: str,
    track_type: str,
    width: float,
    seeds,
    noise_enabled: bool = True,
):
    env = RealWorldHierCurveEnv(
        terrain=terrain,
        corridor_width=width,
        track_type=track_type,
        base_seed=seeds[0],
        noise_enabled=noise_enabled,
    )
    wins, zero_hit_wins = 0, 0
    progs, totals, hits, speeds = [], [], [], []

    for s in seeds:
        obs, _ = env.reset(seed=s)
        state, ep_start, done = None, np.ones((1,), dtype=bool), False
        while not done:
            act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = env.step(int(act))
            done = term or trunc

        wins += int(info["is_success"])
        if info["is_success"] and info["wall_hits"] == 0:
            zero_hit_wins += 1
        progs.append(info["progress_m"])
        totals.append(info["total_length"])
        hits.append(info["wall_hits"])
        elapsed_s = max(0.1, env.steps * 0.1)
        speeds.append(info["progress_m"] / elapsed_s)

    n = len(seeds)
    print(
        f"{label:26s} | solve: {wins:2d}/{n:<2d} (0-hit: {zero_hit_wins:2d}) "
        f"| prog: {np.mean(progs):5.2f}/{np.mean(totals):5.2f} m "
        f"| speed: {np.mean(speeds):.2f} m/s | hits: {np.mean(hits):4.1f}"
    )
    return wins, n


def run_full_suite(ckpt: str, noise_enabled: bool = True):
    model = RecurrentPPO.load(ckpt)
    noise_tag = "NOISE ON (Sim-to-Real)" if noise_enabled else "NOISE OFF (Clean)"
    print(f"\n=== UNSEEN CAPABILITY SUITE | {ckpt} | {noise_tag} ===")

    total_w, total_n = 0, 0

    print("\n--- 1. Unseen Random Curves (Seeds 8000-8009, width=0.44m) ---")
    for t in ("sand", "gravel", "pavement", "ice", "mud"):
        tag = f"unseen_{t}" + (" [OOD]" if t == "mud" else "")
        w, n = eval_block(model, tag, t, "random", 0.44, range(8000, 8010), noise_enabled)
        total_w += w
        total_n += n

    print("\n--- 2. Unseen Tight Corridor (Seeds 8100-8109, width=0.36m) ---")
    for t in ("sand", "gravel", "pavement", "ice"):
        w, n = eval_block(model, f"tight_036m_{t}", t, "random", 0.36, range(8100, 8110), noise_enabled)
        total_w += w
        total_n += n

    print("\n--- 3. Zero-Shot OOD Geometries (Never Seen in Training) ---")
    ood_tests = [
        ("hairpin_180_pavement", "pavement", "hairpin",  0.44, range(8200, 8206)),
        ("hairpin_180_sand",     "sand",     "hairpin",  0.44, range(8200, 8206)),
        ("hairpin_180_ice",      "ice",      "hairpin",  0.44, range(8200, 8206)),
        ("chicane_slalom_sand",  "sand",     "chicane",  0.44, range(8300, 8306)),
        ("chicane_slalom_ice",   "ice",      "chicane",  0.44, range(8300, 8306)),
        ("pinch_048_to_033_sand","sand",     "pinch",    0.44, range(8400, 8406)),
        ("pinch_048_to_033_ice", "ice",      "pinch",    0.44, range(8400, 8406)),
        ("marathon_10.5m_sand",  "sand",     "marathon", 0.44, range(8500, 8506)),
        ("marathon_10.5m_ice",   "ice",      "marathon", 0.44, range(8500, 8506)),
    ]
    for label, terrain, ttype, width, seeds in ood_tests:
        w, n = eval_block(model, label, terrain, ttype, width, seeds, noise_enabled)
        total_w += w
        total_n += n

    print(f"\nOVERALL UNSEEN SCORE: {total_w}/{total_n} ({100.0 * total_w / total_n:.1f}%)\n")


def run_viewer(track_type: str, terrain: str, seed: int, ckpt: str, width: float = 0.44):
    model = RecurrentPPO.load(ckpt)
    env = RealWorldHierCurveEnv(
        terrain=terrain,
        corridor_width=width,
        track_type=track_type,
        base_seed=seed,
        noise_enabled=True,
        render_realtime=True,
    )
    obs, info = env.reset(seed=seed)
    pts = env.track["centerline"]
    cx = float(0.5 * (pts[:, 0].min() + pts[:, 0].max()))
    cy = float(0.5 * (pts[:, 1].min() + pts[:, 1].max()))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 2.5)

    print(
        f"LIVE | track={track_type} | terrain={terrain} | seed={seed} "
        f"| width={width:.2f}m | length={info['total_length']:.2f}m"
    )

    state, ep_start = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [cx, cy, 0.0]
        viewer.cam.distance = 1.15 * span
        viewer.cam.elevation = -68.0

        while viewer.is_running():
            act, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, step_info = env.step(int(act))
            if term or trunc:
                status = "GOAL REACHED" if step_info["is_success"] else "ENDED"
                avg_v = step_info["progress_m"] / max(0.1, env.steps * 0.1)
                print(
                    f"{status} | steps={env.steps} | progress={step_info['progress_m']:.2f}/{step_info['total_length']:.2f}m "
                    f"| avg_speed={avg_v:.2f} m/s | wall_hits={step_info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--view":
        ttype = sys.argv[2] if len(sys.argv) > 2 else "hairpin"
        terrain = sys.argv[3] if len(sys.argv) > 3 else "sand"
        ckpt = sys.argv[4] if len(sys.argv) > 4 else "checkpoints/hier_curve_champion_98pct.zip"
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 8200
        width = float(sys.argv[6]) if len(sys.argv) > 6 else 0.44
        run_viewer(ttype, terrain, seed, ckpt, width)
    else:
        ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hier_curve_champion_98pct.zip"
        clean = "--clean" in sys.argv
        run_full_suite(ckpt, noise_enabled=not clean)