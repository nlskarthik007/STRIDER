import sys
import math
import numpy as np
import mujoco
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.hier_curve_env import RealWorldHierCurveEnv


def _generate_7x7_grid(seed: int, branching: bool = False):
    rng = np.random.default_rng(seed)
    N = 7
    # Walls between cells: h_walls[r, c] is wall below (r, c); v_walls[r, c] is wall right of (r, c)
    h_walls = np.ones((N, N), dtype=bool)
    v_walls = np.ones((N, N), dtype=bool)
    visited = np.zeros((N, N), dtype=bool)

    if not branching:
        # Randomized self-avoiding winding path from (0,0) to (6,6) (no dead ends)
        while True:
            visited[:] = False
            h_walls[:] = True
            v_walls[:] = True
            path = [(0, 0)]
            visited[0, 0] = True
            stuck = False
            while path[-1] != (N - 1, N - 1):
                r, c = path[-1]
                nbrs = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N and not visited[nr, nc]:
                        nbrs.append((nr, nc))
                if not nbrs:
                    stuck = True
                    break
                # Slight bias toward bottom-right so paths stay ~22-36 cells (10-16 m)
                weights = np.array([1.35 if (nr >= r and nc >= c) else 1.0 for nr, nc in nbrs])
                nr, nc = nbrs[int(rng.choice(len(nbrs), p=weights / weights.sum()))]
                if nr == r + 1:
                    h_walls[r, c] = False
                elif nr == r - 1:
                    h_walls[nr, nc] = False
                elif nc == c + 1:
                    v_walls[r, c] = False
                elif nc == c - 1:
                    v_walls[nr, nc] = False
                visited[nr, nc] = True
                path.append((nr, nc))
            if not stuck and len(path) >= 18:
                return h_walls, v_walls, path
    else:
        # Full recursive-backtracker 7x7 maze (includes dead ends & T-junctions)
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            r, c = stack[-1]
            nbrs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and not visited[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                stack.pop()
                continue
            nr, nc = nbrs[int(rng.integers(0, len(nbrs)))]
            if nr == r + 1:
                h_walls[r, c] = False
            elif nr == r - 1:
                h_walls[nr, nc] = False
            elif nc == c + 1:
                v_walls[r, c] = False
            elif nc == c - 1:
                v_walls[nr, nc] = False
            visited[nr, nc] = True
            stack.append((nr, nc))
        return h_walls, v_walls, None


def _shortest_cell_path(h_walls, v_walls, start=(0, 0), goal=(6, 6)):
    N = 7
    q = [(start, [start])]
    seen = {start}
    while q:
        (r, c), p = q.pop(0)
        if (r, c) == goal:
            return p
        moves = []
        if r + 1 < N and not h_walls[r, c]:
            moves.append((r + 1, c))
        if r - 1 >= 0 and not h_walls[r - 1, c]:
            moves.append((r - 1, c))
        if c + 1 < N and not v_walls[r, c]:
            moves.append((r, c + 1))
        if c - 1 >= 0 and not v_walls[r, c - 1]:
            moves.append((r, c - 1))
        for nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, p + [nxt]))
    return [start, goal]


def build_7x7_maze_scene(buggy_xml_path: str, seed: int, cell_size: float = 0.44, branching: bool = False):
    N = 7
    wall_h = 0.14
    half_t = 0.012
    h_walls, v_walls, path = _generate_7x7_grid(seed, branching=branching)
    if path is None:
        path = _shortest_cell_path(h_walls, v_walls)

    # Start cell (0,0) is centered at (0,0); row -> +x, col -> +y
    geoms = []
    # Outer boundary walls
    span = N * cell_size
    c_mid = 0.5 * (N - 1) * cell_size
    geoms.append(f'<geom name="wall_outer_n" type="box" pos="{-0.5*cell_size:.4f} {c_mid:.4f} {0.5*wall_h:.4f}" size="{half_t} {0.5*span+half_t:.4f} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>')
    geoms.append(f'<geom name="wall_outer_s" type="box" pos="{(N-0.5)*cell_size:.4f} {c_mid:.4f} {0.5*wall_h:.4f}" size="{half_t} {0.5*span+half_t:.4f} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>')
    geoms.append(f'<geom name="wall_outer_w" type="box" pos="{c_mid:.4f} {-0.5*cell_size:.4f} {0.5*wall_h:.4f}" size="{0.5*span+half_t:.4f} {half_t} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>')
    geoms.append(f'<geom name="wall_outer_e" type="box" pos="{c_mid:.4f} {(N-0.5)*cell_size:.4f} {0.5*wall_h:.4f}" size="{0.5*span+half_t:.4f} {half_t} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>')

    # Interior walls
    for r in range(N):
        for c in range(N):
            cx, cy = r * cell_size, c * cell_size
            if r < N - 1 and h_walls[r, c]:
                geoms.append(
                    f'<geom name="wall_h_{r}_{c}" type="box" pos="{cx+0.5*cell_size:.4f} {cy:.4f} {0.5*wall_h:.4f}" '
                    f'size="{half_t} {0.5*cell_size+half_t:.4f} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>'
                )
            if c < N - 1 and v_walls[r, c]:
                geoms.append(
                    f'<geom name="wall_v_{r}_{c}" type="box" pos="{cx:.4f} {cy+0.5*cell_size:.4f} {0.5*wall_h:.4f}" '
                    f'size="{0.5*cell_size+half_t:.4f} {half_t} {0.5*wall_h:.4f}" rgba="0.22 0.28 0.36 1"/>'
                )

    # Subdivide cell centerline for accurate arc-length progress tracking
    raw_pts = np.array([(r * cell_size, c * cell_size) for r, c in path], dtype=np.float64)
    dense = [raw_pts[0]]
    for i in range(len(raw_pts) - 1):
        for a in (0.33, 0.67, 1.0):
            dense.append((1.0 - a) * raw_pts[i] + a * raw_pts[i + 1])
    pts = np.array(dense, dtype=np.float64)
    seg_lens = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))

    # Initial orientation facing the first open neighbor
    init_yaw = 0.0 if (len(path) > 1 and path[1][0] == 1) else (0.5 * math.pi)
    qw, qz = math.cos(0.5 * init_yaw), math.sin(0.5 * init_yaw)
    gx, gy = pts[-1]

    scene_xml = f"""
    <mujoco model="maze_7x7">
      <option timestep="0.005" integrator="RK4"/>
      <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="{c_mid:.2f} {c_mid:.2f} 6.0" dir="0 0 -1"/>
        <geom name="terrain_floor" type="plane" size="25 25 0.1"
              rgba="0.78 0.66 0.46 1" friction="0.48 0.02 0.015"
              solref="0.045 0.75" solimp="0.75 0.92 0.015"/>
        {''.join(geoms)}
        <geom name="goal_outer" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.002" size="0.13 0.002" contype="0" conaffinity="0" rgba="0.06 0.72 0.50 0.85"/>
        <geom name="goal_inner" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.003" size="0.06 0.002" contype="0" conaffinity="0" rgba="1.0 1.0 1.0 0.95"/>
      </worldbody>
    </mujoco>
    """

    spec = mujoco.MjSpec.from_string(scene_xml)
    bot_spec = mujoco.MjSpec.from_file(buggy_xml_path)
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.035], quat=[qw, 0.0, 0.0, qz])
    if hasattr(spec, "attach"):
        spec.attach(bot_spec, prefix="bot/", frame=frame)
    else:
        frame.attach_body(bot_spec.worldbody.first_body(), "bot/", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, {
        "centerline": pts,
        "half_w": np.full(len(pts), 0.5 * cell_size, dtype=np.float64),
        "cum_len": cum_len,
        "total_length": float(cum_len[-1]),
        "goal_xy": (float(gx), float(gy)),
        "track_type": "maze_7x7",
    }


class Unseen7x7MazeEnv(RealWorldHierCurveEnv):
    def __init__(self, terrain="sand", cell_size=0.44, branching=False, noise_enabled=True, render_realtime=False):
        super().__init__(
            terrain=terrain,
            corridor_width=cell_size,
            max_steps=650,
            noise_enabled=noise_enabled,
            render_realtime=render_realtime,
        )
        self.cell_size = cell_size
        self.branching = branching

    def reset(self, *, seed=None, options=None):
        super(RealWorldHierCurveEnv, self).reset(seed=seed)
        track_seed = int(seed) if seed is not None else int(self.rng.integers(0, 100000))
        self.model, self.data, self.track = build_7x7_maze_scene(
            self.buggy_xml_path, seed=track_seed, cell_size=self.cell_size, branching=self.branching
        )
        self._cache_ids()
        self.apply_terrain(self.terrain_mode)
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_act = 0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {"total_length": self.track["total_length"]}


def run_bench(ckpt: str, branching: bool = False):
    model = RecurrentPPO.load(ckpt)
    mode_name = "7x7 BRANCHING MAZE (Dead Ends)" if branching else "7x7 WINDING MAZE (Single Path)"
    print(f"\n=== ZERO-SHOT {mode_name} | {ckpt} | NOISE ON ===")
    for terrain in ("pavement", "sand", "gravel", "ice", "mud"):
        env = Unseen7x7MazeEnv(terrain=terrain, cell_size=0.44, branching=branching, noise_enabled=True)
        wins, zero_hits, progs, totals, hits = 0, 0, [], [], []
        for s in range(9000, 9010):
            obs, _ = env.reset(seed=s)
            st, ep, done = None, np.ones((1,), dtype=bool), False
            while not done:
                a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
                ep = np.zeros((1,), dtype=bool)
                obs, _, tm, tc, info = env.step(int(a))
                done = tm or tc
            wins += int(info["is_success"])
            zero_hits += int(info["is_success"] and info["wall_hits"] == 0)
            progs.append(info["progress_m"])
            totals.append(info["total_length"])
            hits.append(info["wall_hits"])
        print(
            f"7x7_{terrain:10s} | solve: {wins:2d}/10 (0-hit: {zero_hits:2d}) "
            f"| prog: {np.mean(progs):5.2f}/{np.mean(totals):5.2f} m | hits: {np.mean(hits):4.1f}"
        )


def run_view(ckpt: str, terrain: str = "sand", seed: int = 9000, branching: bool = False):
    model = RecurrentPPO.load(ckpt)
    env = Unseen7x7MazeEnv(terrain=terrain, cell_size=0.44, branching=branching, noise_enabled=True, render_realtime=True)
    obs, info = env.reset(seed=seed)
    print(f"LIVE 7x7 MAZE | terrain={terrain} | seed={seed} | branching={branching} | path_len={info['total_length']:.2f}m")
    st, ep = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [1.32, 1.32, 0.0]
        viewer.cam.distance = 4.2
        viewer.cam.elevation = -72.0
        while viewer.is_running():
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, step_info = env.step(int(a))
            if tm or tc:
                status = "GOAL REACHED" if step_info["is_success"] else "ENDED"
                print(
                    f"{status} | steps={env.steps} | prog={step_info['progress_m']:.2f}/{step_info['total_length']:.2f}m "
                    f"| wall_hits={step_info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    ckpt = "checkpoints/hier_curve_champion_98pct.zip"
    branching = "--branching" in sys.argv
    if "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a not in ("--view", "--branching")]
        terrain = args[0] if len(args) > 0 else "sand"
        seed = int(args[1]) if len(args) > 1 else 9000
        run_view(ckpt, terrain=terrain, seed=seed, branching=branching)
    else:
        run_bench(ckpt, branching=branching)
