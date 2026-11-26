import math
import time
import numpy as np
from gymnasium import spaces
import mujoco

from env.hier_curve_env import RealWorldHierCurveEnv
from sim.curve_scene_builder import build_curved_track_scene


# (v_nom_mps, curvature_or_yaw_rate, is_in_place_spin)
MAZE_ARC_PRIMITIVES = (
    (0.52,  0.00, False),  # 0: straight fast
    (0.42,  1.15, False),  # 1: gentle left arc
    (0.42, -1.15, False),  # 2: gentle right arc
    (0.26,  2.50, False),  # 3: sharp left arc
    (0.26, -2.50, False),  # 4: sharp right arc
    (0.00,  2.40, True),   # 5: in-place Gyro-PI pivot left (CCW)
    (0.00, -2.40, True),   # 6: in-place Gyro-PI pivot right (CW)
)

# Cardinal directions in MuJoCo world frame: row -> +x, col -> +y
# +1 mod 4 = CCW (Left), -1 mod 4 = CW (Right)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _generate_branching_grid(seed: int, grid_size: int = 5):
    rng = np.random.default_rng(seed)
    N = grid_size
    h_walls = np.ones((N, N), dtype=bool)
    v_walls = np.ones((N, N), dtype=bool)
    visited = np.zeros((N, N), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in DIRS:
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

    return h_walls, v_walls


def is_dir_open(h_walls, v_walls, r: int, c: int, d: int) -> bool:
    N = h_walls.shape[0]
    if d == 0:
        return (r + 1 < N) and (not h_walls[r, c])
    if d == 1:
        return (c + 1 < N) and (not v_walls[r, c])
    if d == 2:
        return (r - 1 >= 0) and (not h_walls[r - 1, c])
    return (c - 1 >= 0) and (not v_walls[r, c - 1])


def _shortest_path(h_walls, v_walls):
    N = h_walls.shape[0]
    goal = (N - 1, N - 1)
    q = [((0, 0), [(0, 0)])]
    seen = {(0, 0)}
    while q:
        (r, c), p = q.pop(0)
        if (r, c) == goal:
            return p
        for d, (dr, dc) in enumerate(DIRS):
            if is_dir_open(h_walls, v_walls, r, c, d):
                nxt = (r + dr, c + dc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, p + [nxt]))
    return [(0, 0), goal]


def build_terrain_maze_scene(
    buggy_xml_path: str,
    seed: int = 0,
    grid_size: int = 5,
    cell_size: float = 0.44,
    wall_height: float = 0.14,
):
    N = grid_size
    half_t = 0.012
    h_walls, v_walls = _generate_branching_grid(seed, grid_size=N)
    bfs_path = _shortest_path(h_walls, v_walls)

    geoms = []
    span = N * cell_size
    c_mid = 0.5 * (N - 1) * cell_size
    geoms.append(
        f'<geom name="wall_outer_n" type="box" pos="{-0.5*cell_size:.4f} {c_mid:.4f} {0.5*wall_height:.4f}" '
        f'size="{half_t} {0.5*span+half_t:.4f} {0.5*wall_height:.4f}" rgba="0.22 0.28 0.36 1"/>'
    )
    geoms.append(
        f'<geom name="wall_outer_s" type="box" pos="{(N-0.5)*cell_size:.4f} {c_mid:.4f} {0.5*wall_height:.4f}" '
        f'size="{half_t} {0.5*span+half_t:.4f} {0.5*wall_height:.4f}" rgba="0.22 0.28 0.36 1"/>'
    )
    geoms.append(
        f'<geom name="wall_outer_w" type="box" pos="{c_mid:.4f} {-0.5*cell_size:.4f} {0.5*wall_height:.4f}" '
        f'size="{0.5*span+half_t:.4f} {half_t} {0.5*wall_height:.4f}" rgba="0.22 0.28 0.36 1"/>'
    )
    geoms.append(
        f'<geom name="wall_outer_e" type="box" pos="{c_mid:.4f} {(N-0.5)*cell_size:.4f} {0.5*wall_height:.4f}" '
        f'size="{0.5*span+half_t:.4f} {half_t} {0.5*wall_height:.4f}" rgba="0.22 0.28 0.36 1"/>'
    )

    for r in range(N):
        for c in range(N):
            cx, cy = r * cell_size, c * cell_size
            if r < N - 1 and h_walls[r, c]:
                geoms.append(
                    f'<geom name="wall_h_{r}_{c}" type="box" '
                    f'pos="{cx+0.5*cell_size:.4f} {cy:.4f} {0.5*wall_height:.4f}" '
                    f'size="{half_t} {0.5*cell_size+half_t:.4f} {0.5*wall_height:.4f}" '
                    f'rgba="0.22 0.28 0.36 1"/>'
                )
            if c < N - 1 and v_walls[r, c]:
                geoms.append(
                    f'<geom name="wall_v_{r}_{c}" type="box" '
                    f'pos="{cx:.4f} {cy+0.5*cell_size:.4f} {0.5*wall_height:.4f}" '
                    f'size="{0.5*cell_size+half_t:.4f} {half_t} {0.5*wall_height:.4f}" '
                    f'rgba="0.22 0.28 0.36 1"/>'
                )

    raw_pts = np.array([(r * cell_size, c * cell_size) for r, c in bfs_path], dtype=np.float64)
    dense = [raw_pts[0]]
    for i in range(len(raw_pts) - 1):
        for a in (0.33, 0.67, 1.0):
            dense.append((1.0 - a) * raw_pts[i] + a * raw_pts[i + 1])
    pts = np.array(dense, dtype=np.float64)
    seg_lens = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))

    init_dir = 0 if is_dir_open(h_walls, v_walls, 0, 0, 0) else 1
    init_yaw = 0.0 if init_dir == 0 else (0.5 * math.pi)
    qw, qz = math.cos(0.5 * init_yaw), math.sin(0.5 * init_yaw)
    gx, gy = (N - 1) * cell_size, (N - 1) * cell_size

    scene_xml = f"""
    <mujoco model="terrain_maze_{N}x{N}">
      <option timestep="0.005" integrator="RK4"/>
      <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="{c_mid:.2f} {c_mid:.2f} 6.0" dir="0 0 -1"/>
        <geom name="terrain_floor" type="plane" size="25 25 0.1"
              rgba="0.78 0.66 0.46 1" friction="0.48 0.02 0.015"
              solref="0.045 0.75" solimp="0.75 0.92 0.015"/>
        {''.join(geoms)}
        <geom name="goal_outer" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.002"
              size="0.13 0.002" contype="0" conaffinity="0" rgba="0.06 0.72 0.50 0.85"/>
        <geom name="goal_inner" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.003"
              size="0.06 0.002" contype="0" conaffinity="0" rgba="1.0 1.0 1.0 0.95"/>
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
        "track_type": "maze",
        "grid_size": N,
        "cell_size": cell_size,
        "h_walls": h_walls,
        "v_walls": v_walls,
        "init_dir": init_dir,
    }


class RealWorldTerrainMazeEnv(RealWorldHierCurveEnv):
    """Unified 7-action environment supporting branching terrain mazes AND continuous curves."""

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        mode: str = "maze",          # "maze", "curve", or "mixed"
        grid_size=7,                 # int or tuple e.g. (5, 6)
        terrain: str = "domain_rand",
        corridor_width=0.44,
        track_type: str = "train_mix",
        max_steps: int = None,
        base_seed: int = 0,
        noise_enabled: bool = True,
        render_realtime: bool = False,
    ):
        super().__init__(
            buggy_xml_path=buggy_xml_path,
            terrain=terrain,
            corridor_width=corridor_width,
            track_type=track_type,
            max_steps=max_steps if max_steps is not None else 1200,
            base_seed=base_seed,
            noise_enabled=noise_enabled,
            render_realtime=render_realtime,
        )
        self.mode = mode
        self.grid_size = grid_size
        self.user_max_steps = max_steps
        self.action_space = spaces.Discrete(len(MAZE_ARC_PRIMITIVES))
        self.visited_cells = set()

    def _get_obs(self) -> np.ndarray:
        rf_f = self._read_rf("rf_front")
        rf_b = self._read_rf("rf_back")
        rf_l = self._read_rf("rf_left")
        rf_r = self._read_rf("rf_right")
        rf_fl = self._read_rf("rf_fl")
        rf_fr = self._read_rf("rf_fr")

        wl, wr = self._read_wheel_speeds()
        enc_v = 0.5 * (wl + wr) * self.wheel_radius
        enc_w = (wr - wl) * self.wheel_radius / self.track_width
        gyro_z = self._read_gyro_z()
        yaw_slip = enc_w - gyro_z

        lat_bal = (rf_l - rf_r) / 0.22
        diag_bal = (rf_fl - rf_fr) / 0.28
        corridor_cue = np.clip((rf_l + rf_r) / 0.45, 0.0, 1.0)

        prev_v, prev_k, _ = MAZE_ARC_PRIMITIVES[self._prev_act]

        obs = np.array(
            [
                rf_f / self.max_range,
                rf_b / self.max_range,
                rf_l / self.max_range,
                rf_r / self.max_range,
                rf_fl / self.max_range,
                rf_fr / self.max_range,
                np.clip(enc_v / 0.60, -1.0, 1.0),
                np.clip(gyro_z / 3.0, -1.0, 1.0),
                np.clip(yaw_slip / 3.0, -1.0, 1.0),
                np.clip(lat_bal, -1.0, 1.0),
                np.clip(diag_bal, -1.0, 1.0),
                float(corridor_cue),
                prev_v / 0.55,
                np.clip(prev_k / 2.50, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super(RealWorldHierCurveEnv, self).reset(seed=seed)
        track_seed = int(seed) if seed is not None else int(self.rng.integers(0, 100000))

        use_maze = (self.mode == "maze") or (self.mode == "mixed" and self.rng.random() < 0.75)

        if use_maze:
            N = int(self.rng.choice(self.grid_size)) if isinstance(self.grid_size, (tuple, list)) else int(self.grid_size)
            c_w = float(self.rng.uniform(0.40, 0.46)) if isinstance(self.corridor_width, (tuple, list)) else float(self.corridor_width)
            self.current_corridor_width = c_w
            self.model, self.data, self.track = build_terrain_maze_scene(
                self.buggy_xml_path, seed=track_seed, grid_size=N, cell_size=c_w
            )
            self.max_steps = self.user_max_steps if self.user_max_steps is not None else (750 if N <= 5 else (1050 if N == 6 else 2000))
        else:
            c_w = (
                float(self.rng.uniform(self.corridor_width[0], self.corridor_width[1]))
                if isinstance(self.corridor_width, (tuple, list))
                else float(self.corridor_width)
            )
            self.current_corridor_width = c_w
            self.model, self.data, self.track = build_curved_track_scene(
                self.buggy_xml_path, seed=track_seed, corridor_width=c_w, track_type=self.track_type
            )
            self.max_steps = self.user_max_steps if self.user_max_steps is not None else (420 if self.track_type == "marathon" else 280)

        self._cache_ids()

        if self.terrain_mode == "domain_rand":
            if self.rng.random() < 0.40:
                t_choice = "continuous_dr"
            else:
                t_choice = str(self.rng.choice(["sand", "gravel", "pavement", "ice"], p=[0.30, 0.20, 0.20, 0.30]))
        elif self.terrain_mode == "random":
            t_choice = str(self.rng.choice(["sand", "pavement", "gravel", "ice"], p=[0.35, 0.15, 0.15, 0.35]))
        else:
            t_choice = self.terrain_mode
        self.apply_terrain(t_choice)

        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_act = 0
        self.visited_cells = {(0, 0)}

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {
            "track_seed": track_seed,
            "terrain": self.current_terrain,
            "track_type": self.track["track_type"],
            "total_length": self.track["total_length"],
        }

    def step(self, action):
        act = int(np.asarray(action).item())
        self.steps += 1

        v_nom, k_or_w, is_spin = MAZE_ARC_PRIMITIVES[act]
        dt = self.model.opt.timestep
        max_dv = 3.2* dt
        max_dw = 16.0 * dt
        step_hits = 0

        for _ in range(self.ticks_per_step):
            rf_f = self._read_rf("rf_front")
            rf_b = self._read_rf("rf_back")
            rf_l = self._read_rf("rf_left")
            rf_r = self._read_rf("rf_right")
            rf_fl = self._read_rf("rf_fl")
            rf_fr = self._read_rf("rf_fr")

            if is_spin:
                # Front + diagonal clearance centering so chassis corners never scrape during 90/180 pivots
                v_target = 0.0
                front_clear = min(rf_f, rf_fl + 0.015, rf_fr + 0.015)
                if front_clear < 0.125 and rf_b > 0.105:
                    v_target = -0.14
                elif rf_b < 0.105 and front_clear > 0.125:
                    v_target = 0.12
                w_des = k_or_w
            else:
                # Allow full stop right in front of dead-end walls (0.0 floor instead of 0.30)
                front_scale = float(np.clip((rf_f - 0.115) / 0.22, 0.0, 1.0))
                width_scale = float(np.clip(((rf_l + rf_r) - 0.15) / 0.12, 0.55, 1.0))
                diag_scale = float(np.clip((min(rf_fl, rf_fr) - 0.095) / 0.10, 0.45, 1.0))
                v_target = v_nom * min(front_scale, width_scale, diag_scale)
                w_des = v_target * k_or_w

                repulse = 0.0
                if rf_l < 0.062:
                    repulse -= 8.0 * (0.062 - rf_l)
                if rf_r < 0.062:
                    repulse += 8.0 * (0.062 - rf_r)
                if rf_fl < 0.092:
                    repulse -= 9.0 * (0.092 - rf_fl)
                if rf_fr < 0.092:
                    repulse += 9.0 * (0.092 - rf_fr)
                w_des = float(np.clip(w_des + repulse, -3.5, 3.5))

            # Closed-loop Gyro PI controller (compensates sand/mud drag & ice slip)
            gz = self._read_gyro_z()
            w_err = w_des - gz
            self._w_err_int = float(np.clip(self._w_err_int + w_err * dt, -0.5, 0.5))
            w_target = float(np.clip(w_des + 0.55 * w_err + 1.40 * self._w_err_int, -4.0, 4.0))

            self._cur_v += float(np.clip(v_target - self._cur_v, -max_dv, max_dv))
            self._cur_w += float(np.clip(w_target - self._cur_w, -max_dw, max_dw))

            w_l = (self._cur_v - 0.5 * self._cur_w * self.track_width) / self.wheel_radius
            w_r = (self._cur_v + 0.5 * self._cur_w * self.track_width) / self.wheel_radius
            self.data.ctrl[self._act_l] = float(np.clip(w_l, -self.max_wheel_vel, self.max_wheel_vel))
            self.data.ctrl[self._act_r] = float(np.clip(w_r, -self.max_wheel_vel, self.max_wheel_vel))

            if self.sand_drag > 0.0:
                vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
                wz = float(self.data.qvel[5])
                self.data.qfrc_applied[0] = -self.sand_drag * vx * (1.0 + 2.0 * abs(vx))
                self.data.qfrc_applied[1] = -self.sand_drag * vy * (1.0 + 2.0 * abs(vy))
                self.data.qfrc_applied[5] = -0.08 * self.sand_drag * wz
            else:
                self.data.qfrc_applied[:6] = 0.0

            if self.viewer is not None and hasattr(self.viewer, "lock"):
                with self.viewer.lock():
                    mujoco.mj_step(self.model, self.data)
            else:
                mujoco.mj_step(self.model, self.data)

            if self._count_wall_contacts():
                step_hits += 1
                self.wall_hits += 1

            if self.viewer is not None:
                self.viewer.sync()
                if self.render_realtime:
                    time.sleep(dt)

        obs = self._get_obs()
        gx, gy = self.track["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        at_goal = math.hypot(bx - gx, by - gy) < 0.22

        if self.track["track_type"] == "maze":
            cs = self.track["cell_size"]
            N = self.track["grid_size"]
            cell = (
                int(np.clip(round(bx / cs), 0, N - 1)),
                int(np.clip(round(by / cs), 0, N - 1)),
            )
            new_cell = cell not in self.visited_cells
            self.visited_cells.add(cell)
            reward = (0.6 if new_cell else -0.02)
            progress = float(len(self.visited_cells))
        else:
            progress, lat_err, v_along, local_half_w = self._track_metrics()
            center_factor = max(0.0, 1.0 - lat_err / max(local_half_w, 0.10))
            reward = 1.40 * v_along * (0.35 + 0.65 * center_factor) - 0.04

        if act != self._prev_act:
            reward -= 0.015

        if step_hits > 0:
            self._consec_hits += 1
            reward -= 1.2
        else:
            self._consec_hits = 0

        # Allow slightly higher hit budget on long 7x7 backtracking mazes before crash termination
        hit_cap = 35 if self.track["track_type"] == "maze" else 25
        crashed = (self._consec_hits >= 4) or (self.wall_hits >= hit_cap)
        if crashed:
            reward -= 8.0
        if at_goal:
            reward += 25.0 + (8.0 if self.wall_hits == 0 else 0.0)

        self._prev_act = act
        terminated = at_goal or crashed
        truncated = (not terminated) and (self.steps >= self.max_steps)

        info = {
            "is_success": at_goal,
            "crashed": crashed,
            "terrain": self.current_terrain,
            "wall_hits": self.wall_hits,
            "visited_cells": len(self.visited_cells),
            "progress_m": progress,
            "total_length": self.track["total_length"],
        }
        return obs, float(reward), terminated, truncated, info
