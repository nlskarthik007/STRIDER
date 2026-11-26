import math
import time
from collections import deque
import numpy as np
from gymnasium import spaces
import mujoco

from env.hier_curve_env import RealWorldHierCurveEnv
from env.hier_maze_terrain_env import build_terrain_maze_scene


DIRS_4 = ((1, 0), (0, 1), (-1, 0), (0, -1))  # 0:E (+X), 1:N (+Y), 2:W (-X), 3:S (-Y)


class RealWorldContinuousEuclideanMazeEnv(RealWorldHierCurveEnv):
    """
    Continuous-motion (v, w) multi-terrain branching maze environment with
    ego-frame Euclidean distance & bearing observations + 10-dead-end path memory.
    """

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        grid_size=7,
        terrain: str = "domain_rand",
        cell_size=0.44,
        max_steps: int = None,
        base_seed: int = 0,
        noise_enabled: bool = True,
        render_realtime: bool = False,
        max_deadend_memory: int = 10,
    ):
        super().__init__(
            buggy_xml_path=buggy_xml_path,
            terrain=terrain,
            corridor_width=cell_size,
            track_type="maze",
            max_steps=max_steps if max_steps is not None else 1200,
            base_seed=base_seed,
            noise_enabled=noise_enabled,
            render_realtime=render_realtime,
        )
        self.grid_size = grid_size
        self.cell_size_cfg = cell_size
        self.user_max_steps = max_steps
        self.max_deadend_memory = max_deadend_memory

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.visited_cells = set()
        self.path_stack = []
        self.dead_end_paths = deque(maxlen=max_deadend_memory)
        self.sealed_entries = deque(maxlen=max_deadend_memory * 4)
        self.dead_cells = set()
        self._in_dead_end_pocket = False
        self._prev_euc_dist = 0.0
        self._prev_w_norm = 0.0

    def _body_pose(self):
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return bx, by, yaw

    def _cell_coords(self, bx: float, by: float):
        cs = self.track["cell_size"]
        N = self.track["grid_size"]
        # Works for both 0-centered cells (x*cs) and half-offset cells ((x+0.5)*cs)
        gx, _ = self.track["goal_xy"]
        offset = 0.5 * cs if abs(gx - (N - 0.5) * cs) < 1e-3 else 0.0
        cx = int(np.clip(round((bx - offset) / cs), 0, N - 1))
        cy = int(np.clip(round((by - offset) / cs), 0, N - 1))
        return cx, cy, offset

    def _open_dirs_from_local_tof(self, yaw: float):
        """Infer open cardinal directions around current cell purely from noisy 6x ToF sensors."""
        cs = self.track["cell_size"]
        thresh = 0.62 * cs
        cur_dir = int(round(yaw / (0.5 * math.pi))) % 4
        rf_map = {
            cur_dir: self._read_rf("rf_front"),
            (cur_dir + 1) % 4: self._read_rf("rf_left"),
            (cur_dir + 2) % 4: self._read_rf("rf_back"),
            (cur_dir + 3) % 4: self._read_rf("rf_right"),
        }
        return [d for d, dist in rf_map.items() if dist > thresh], rf_map

    def _update_deadend_memory(self):
        bx, by, yaw = self._body_pose()
        cell = self._cell_coords(bx, by)[:2]
        open_dirs, _ = self._open_dirs_from_local_tof(yaw)

        if not self.path_stack:
            self.path_stack = [cell]
            return

        last_cell = self.path_stack[-1]
        if cell != last_cell:
            if cell in self.path_stack:
                # Buggy backtracked into an earlier cell on its active path -> seal popped spur
                idx = self.path_stack.index(cell)
                spur = self.path_stack[idx:]  # starts at junction `cell`, followed by dead-end cells
                if len(spur) >= 2:
                    dx = spur[1][0] - spur[0][0]
                    dy = spur[1][1] - spur[0][1]
                    if (dx, dy) in DIRS_4:
                        entry = (spur[0], DIRS_4.index((dx, dy)))
                        if entry not in self.sealed_entries:
                            self.sealed_entries.append(entry)
                            self.dead_end_paths.append(tuple(spur))
                            for dead_c in spur[1:]:
                                self.dead_cells.add(dead_c)
                self.path_stack = self.path_stack[: idx + 1]
                self._in_dead_end_pocket = False
            else:
                self.path_stack.append(cell)

        # Detect dead-end pocket (<= 1 open exit in current cell)
        if len(open_dirs) <= 1 and len(self.path_stack) >= 2:
            self._in_dead_end_pocket = True

        # Recursive junction roll-up: if every open exit at current cell is sealed or dead, mark this cell dead too
        unsealed = [
            d for d in open_dirs
            if (cell, d) not in self.sealed_entries
            and (cell[0] + DIRS_4[d][0], cell[1] + DIRS_4[d][1]) not in self.dead_cells
        ]
        if len(unsealed) == 0 and len(self.path_stack) >= 2:
            parent = self.path_stack[-2]
            dx, dy = cell[0] - parent[0], cell[1] - parent[1]
            if (dx, dy) in DIRS_4:
                entry = (parent, DIRS_4.index((dx, dy)))
                if entry not in self.sealed_entries:
                    self.sealed_entries.append(entry)
                    self.dead_end_paths.append((parent, cell))
                self.dead_cells.add(cell)

    def _euclidean_goal_features(self):
        bx, by, yaw = self._body_pose()
        gx, gy = self.track["goal_xy"]
        dx, dy = gx - bx, gy - by
        euc_dist = math.hypot(dx, dy)
        max_diag = math.hypot(gx, gy) + 1e-6

        # Check if direct Euclidean bearing points into any of the up-to-10 remembered dead-end paths
        cs = self.track["cell_size"]
        cx, cy, offset = self._cell_coords(bx, by)
        cell = (cx, cy)
        open_dirs, _ = self._open_dirs_from_local_tof(yaw)

        target_x, target_y = gx, gy
        if open_dirs:
            unsealed = [
                d for d in open_dirs
                if (cell, d) not in self.sealed_entries
                and (cx + DIRS_4[d][0], cy + DIRS_4[d][1]) not in self.dead_cells
            ]
            # If we are at a junction with sealed branches, steer Euclidean guide toward best unsealed exit
            if unsealed and len(unsealed) < len(open_dirs):
                best_d = min(
                    unsealed,
                    key=lambda d: math.hypot(
                        ((cx + DIRS_4[d][0]) * cs + offset) - gx,
                        ((cy + DIRS_4[d][1]) * cs + offset) - gy,
                    ),
                )
                target_x = (cx + DIRS_4[best_d][0]) * cs + offset
                target_y = (cy + DIRS_4[best_d][1]) * cs + offset
            elif not unsealed and len(self.path_stack) >= 2:
                # All forward branches dead -> point Euclidean guide back along backtrack parent
                px, py = self.path_stack[-2]
                target_x = px * cs + offset
                target_y = py * cs + offset

        goal_world_ang = math.atan2(target_y - by, target_x - bx)
        rel_ang = (goal_world_ang - yaw + math.pi) % (2.0 * math.pi) - math.pi

        if self.noise_enabled:
            euc_dist_obs = max(0.0, euc_dist + float(self.rng.normal(0.0, 0.015)))
            rel_ang_obs = rel_ang + float(self.rng.normal(0.0, 0.02))
        else:
            euc_dist_obs = euc_dist
            rel_ang_obs = rel_ang

        return (
            euc_dist,
            float(np.clip(euc_dist_obs / max_diag, 0.0, 1.5)),
            math.cos(rel_ang_obs),
            math.sin(rel_ang_obs),
        )

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
        _, euc_norm, g_cos, g_sin = self._euclidean_goal_features()

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
                float(np.clip(euc_norm, 0.0, 1.0)),
                float(g_cos),
                float(g_sin),
                float(self._prev_w_norm),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super(RealWorldHierCurveEnv, self).reset(seed=seed)
        track_seed = int(seed) if seed is not None else int(self.rng.integers(0, 100000))

        N = (
            int(self.rng.choice(self.grid_size))
            if isinstance(self.grid_size, (tuple, list))
            else int(self.grid_size)
        )
        cs = (
            float(self.rng.uniform(self.cell_size_cfg[0], self.cell_size_cfg[1]))
            if isinstance(self.cell_size_cfg, (tuple, list))
            else float(self.cell_size_cfg)
        )
        self.current_corridor_width = cs
        self.model, self.data, self.track = build_terrain_maze_scene(
            self.buggy_xml_path, seed=track_seed, grid_size=N, cell_size=cs
        )
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

        self.max_steps = (
            self.user_max_steps
            if self.user_max_steps is not None
            else (600 if N <= 5 else (900 if N == 6 else 1300))
        )
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_w_norm = 0.0
        self.visited_cells = {(0, 0)}
        self.path_stack = [(0, 0)]
        self.dead_end_paths = deque(maxlen=self.max_deadend_memory)
        self.sealed_entries = deque(maxlen=self.max_deadend_memory * 4)
        self.dead_cells = set()
        self._in_dead_end_pocket = False

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        euc_dist, _, _, _ = self._euclidean_goal_features()
        self._prev_euc_dist = euc_dist

        return self._get_obs(), {
            "track_seed": track_seed,
            "terrain": self.current_terrain,
            "grid_size": N,
            "euc_dist": euc_dist,
        }

    def step(self, action):
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        v_norm = float(np.clip(act[0], -1.0, 1.0))
        w_norm = float(np.clip(act[1], -1.0, 1.0))

        v_cmd = 0.5 * (v_norm + 1.0) * 0.48
        w_cmd = w_norm * 2.20
        is_pivot = (v_cmd < 0.04) and (abs(w_cmd) > 0.65)

        self.steps += 1
        dt = self.model.opt.timestep
        max_dv = 2.6 * dt
        max_dw = 14.0 * dt
        step_hits = 0

        for _ in range(self.ticks_per_step):
            rf_f = self._read_rf("rf_front")
            rf_b = self._read_rf("rf_back")
            rf_l = self._read_rf("rf_left")
            rf_r = self._read_rf("rf_right")
            rf_fl = self._read_rf("rf_fl")
            rf_fr = self._read_rf("rf_fr")

            if is_pivot:
                v_target = 0.0
                front_clear = min(rf_f, rf_fl + 0.015, rf_fr + 0.015)
                if front_clear < 0.125 and rf_b > 0.105:
                    v_target = -0.12
                elif rf_b < 0.105 and front_clear > 0.125:
                    v_target = 0.10
                w_des = w_cmd
            else:
                front_scale = float(np.clip((rf_f - 0.120) / 0.28, 0.0, 1.0))
                diag_scale = float(np.clip((min(rf_fl, rf_fr) - 0.090) / 0.10, 0.35, 1.0))
                v_target = v_cmd * min(front_scale, diag_scale)

                repulse = 0.0
                if rf_l < 0.062:
                    repulse -= 8.0 * (0.062 - rf_l)
                if rf_r < 0.062:
                    repulse += 8.0 * (0.062 - rf_r)
                if rf_fl < 0.092:
                    repulse -= 9.0 * (0.092 - rf_fl)
                if rf_fr < 0.092:
                    repulse += 9.0 * (0.092 - rf_fr)
                w_des = float(np.clip(w_cmd + repulse, -3.2, 3.2))

            gz = self._read_gyro_z()
            w_err = w_des - gz
            self._w_err_int = float(np.clip(self._w_err_int + w_err * dt, -0.5, 0.5))
            w_target = float(np.clip(w_des + 0.60 * w_err + 1.60 * self._w_err_int, -3.8, 3.8))

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

        self._update_deadend_memory()
        obs = self._get_obs()
        euc_dist, _, _, _ = self._euclidean_goal_features()
        at_goal = euc_dist < 0.22

        bx, by, _ = self._body_pose()
        cell = self._cell_coords(bx, by)[:2]
        new_cell = cell not in self.visited_cells
        self.visited_cells.add(cell)

        euc_delta = self._prev_euc_dist - euc_dist
        self._prev_euc_dist = euc_dist
        reward = 3.5 * euc_delta + (0.5 if new_cell else -0.015)

        if step_hits > 0:
            self._consec_hits += 1
            reward -= 1.2
        else:
            self._consec_hits = 0

        crashed = (self._consec_hits >= 5) or (self.wall_hits >= 30)
        if crashed:
            reward -= 10.0
        if at_goal:
            reward += 30.0 + (10.0 if self.wall_hits == 0 else 0.0)

        self._prev_w_norm = w_norm
        terminated = at_goal or crashed
        truncated = (not terminated) and (self.steps >= self.max_steps)

        info = {
            "is_success": at_goal,
            "crashed": crashed,
            "terrain": self.current_terrain,
            "wall_hits": self.wall_hits,
            "visited_cells": len(self.visited_cells),
            "euc_dist": euc_dist,
            "deadends_remembered": len(self.dead_end_paths),
        }
        return obs, float(reward), terminated, truncated, info
