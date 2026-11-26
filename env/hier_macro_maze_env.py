import math
import time
import numpy as np
from gymnasium import spaces
import mujoco

from env.hier_curve_env import RealWorldHierCurveEnv
from env.hier_maze_terrain_env import build_terrain_maze_scene, DIRS


# Macro skill options: delta_yaw (rad) executed via unwrapped Gyro-PI, then 1-cell advance
MACRO_TURN_DELTAS = (
    0.0,              # 0: Forward 1 cell
    -0.5 * math.pi,   # 1: Turn Right 90 deg + Forward 1 cell
    0.5 * math.pi,    # 2: Turn Left 90 deg + Forward 1 cell
    math.pi,          # 3: U-Turn 180 deg (dead-end backtrack) + Forward 1 cell
)


class RealWorldMacroMazeEnv(RealWorldHierCurveEnv):
    """
    Hierarchical Terrain Maze Environment (Discrete(4) Options over 200 Hz Gyro-PI + ToF skills).
    100% local noisy onboard sensors; handles pavement, gravel, sand, ice (mu=0.08), and OOD mud.
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
    ):
        super().__init__(
            buggy_xml_path=buggy_xml_path,
            terrain=terrain,
            corridor_width=cell_size,
            track_type="maze",
            max_steps=max_steps if max_steps is not None else 120,
            base_seed=base_seed,
            noise_enabled=noise_enabled,
            render_realtime=render_realtime,
        )
        self.grid_size = grid_size
        self.cell_size_cfg = cell_size
        self.user_max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self._est_yaw = 0.0
        self._cardinal_yaw = 0.0
        self._center_fb = 0.133
        self._center_lr = 0.133
        self._last_slip = 0.0
        self.visited_cells = set()

    def _get_obs(self) -> np.ndarray:
        rf_f = self._read_rf("rf_front")
        rf_b = self._read_rf("rf_back")
        rf_l = self._read_rf("rf_left")
        rf_r = self._read_rf("rf_right")
        rf_fl = self._read_rf("rf_fl")
        rf_fr = self._read_rf("rf_fr")

        # Smooth open-corridor cues computed purely from noisy local ToF
        open_f = np.clip((rf_f - (self._center_fb + 0.08)) / 0.12, 0.0, 1.0)
        open_r = np.clip((rf_r - (self._center_lr + 0.08)) / 0.12, 0.0, 1.0)
        open_l = np.clip((rf_l - (self._center_lr + 0.08)) / 0.12, 0.0, 1.0)
        open_b = np.clip((rf_b - (self._center_fb + 0.08)) / 0.12, 0.0, 1.0)

        obs = np.array(
            [
                rf_f / self.max_range,
                rf_b / self.max_range,
                rf_l / self.max_range,
                rf_r / self.max_range,
                rf_fl / self.max_range,
                rf_fr / self.max_range,
                float(open_f),
                float(open_r),
                float(open_l),
                float(open_b),
                float(np.clip(self._last_slip / 2.0, -1.0, 1.0)),
                float(self._prev_act) / 3.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _physics_tick(self, v_target: float, w_des: float, max_dv: float, max_dw: float):
        dt = self.model.opt.timestep
        gz = self._read_gyro_z()
        self._est_yaw += gz * dt

        w_err = w_des - gz
        self._w_err_int = float(np.clip(self._w_err_int + w_err * dt, -0.5, 0.5))
        w_target = float(np.clip(w_des + 0.60 * w_err + 1.80 * self._w_err_int, -3.8, 3.8))

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

        hit = self._count_wall_contacts()
        if hit:
            self.wall_hits += 1

        if self.viewer is not None:
            self.viewer.sync()
            if self.render_realtime:
                time.sleep(dt)

        wl, wr = self._read_wheel_speeds()
        enc_v = 0.5 * (wl + wr) * self.wheel_radius
        enc_w = (wr - wl) * self.wheel_radius / self.track_width
        self._last_slip = enc_w - gz
        return hit, enc_v, gz

    def _exec_turn(self, delta_yaw: float) -> int:
        if abs(delta_yaw) < 1e-3:
            return 0

        # Unwrapped cardinal target cancels prior drift and eliminates ±pi wrap flips
        target_yaw = round((self._est_yaw + delta_yaw) / (0.5 * math.pi)) * (0.5 * math.pi)
        self._cardinal_yaw = target_yaw
        self._w_err_int = 0.0

        dt = self.model.opt.timestep
        max_ticks = 520 if abs(delta_yaw) > 2.0 else 320
        hits = 0
        settled = 0

        for _ in range(max_ticks):
            err = target_yaw - self._est_yaw
            rf_f = self._read_rf("rf_front")
            rf_b = self._read_rf("rf_back")

            # Keep chassis centered inside dead-end pockets during 180-deg pivots
            v_target = 0.0
            if rf_f < self._center_fb - 0.025 and rf_b > self._center_fb - 0.020:
                v_target = -0.08
            elif rf_b < self._center_fb - 0.025 and rf_f > self._center_fb - 0.020:
                v_target = 0.08

            w_des = float(np.clip(3.4 * err, -1.75, 1.75))
            hit, _, gz = self._physics_tick(v_target, w_des, max_dv=1.5 * dt, max_dw=7.5 * dt)
            hits += hit

            if abs(err) < 0.035 and abs(gz) < 0.14:
                settled += 1
                if settled >= 6:
                    break
            else:
                settled = 0

        return hits

    def _exec_forward_one_cell(self) -> int:
        cs = self.track["cell_size"]
        dt = self.model.opt.timestep
        target_yaw = self._cardinal_yaw
        self._w_err_int = 0.0

        rf_f0 = self._read_rf("rf_front")
        if rf_f0 < self._center_fb + 0.11:
            return 0

        # Zero-slip longitudinal laser lock (immune to mud/sand/gravel wheel slip)
        safe_max = 0.85 * self.max_range
        rf_b0 = self._read_rf("rf_back")
        use_front_lock = rf_f0 < safe_max
        use_rear_lock = (not use_front_lock) and (rf_b0 < self._center_fb + 0.10) and (self._center_fb + cs < safe_max)

        if use_front_lock:
            k_f = max(0, int(round((rf_f0 - cs - self._center_fb) / cs)))
            target_rf_f = self._center_fb + k_f * cs + (0.006 if k_f == 0 else 0.0)
        else:
            target_rf_f = self._center_fb + 0.006

        target_rf_b = self._center_fb + cs
        dist_target = cs
        if rf_b0 < self._center_fb + 0.07:
            dist_target = cs - float(np.clip(rf_b0 - self._center_fb, -0.05, 0.05))

        s_enc = 0.0
        hits = 0
        settled = 0
        gx, gy = self.track["goal_xy"]

        for _ in range(800):
            rf_f = self._read_rf("rf_front")
            rf_b = self._read_rf("rf_back")
            rf_l = self._read_rf("rf_left")
            rf_r = self._read_rf("rf_right")
            rf_fl = self._read_rf("rf_fl")
            rf_fr = self._read_rf("rf_fr")

            rem_enc = max(0.0, dist_target - s_enc)
            rem_wall = max(0.0, rf_f - (self._center_fb + 0.006)) if rf_f < 0.48 else 1.0

            if use_front_lock and rf_f < safe_max:
                rem = max(0.0, rf_f - target_rf_f)
            elif use_rear_lock and rf_b < safe_max:
                rem = min(max(0.0, target_rf_b - rf_b), rem_wall)
            elif rf_f < (self._center_fb + 0.16):
                rem = rem_wall
            else:
                rem = min(rem_enc, rem_wall)

            # Traction-safe braking profile (0.40 m/s^2 < mu_ice * g = 0.78 m/s^2)
            if rem <= 0.012:
                v_target = 0.0
            else:
                v_target = min(0.30, math.sqrt(2.0 * 0.40 * rem) + 0.03)

            yaw_err = target_yaw - self._est_yaw
            l_wall = rf_l < (self._center_lr + 0.045)
            r_wall = rf_r < (self._center_lr + 0.045)
            if l_wall and r_wall:
                lat_err = 0.5 * (rf_l - rf_r)
            elif l_wall:
                lat_err = rf_l - self._center_lr
            elif r_wall:
                lat_err = self._center_lr - rf_r
            else:
                lat_err = 0.0

            w_des = 2.8 * yaw_err + 4.2 * float(np.clip(lat_err, -0.05, 0.05))
            if rf_l < 0.058:
                w_des -= 8.0 * (0.058 - rf_l)
            if rf_r < 0.058:
                w_des += 8.0 * (0.058 - rf_r)
            if rf_fl < 0.088:
                w_des -= 7.0 * (0.088 - rf_fl)
            if rf_fr < 0.088:
                w_des += 7.0 * (0.088 - rf_fr)
            w_des = float(np.clip(w_des, -2.2, 2.2))

            hit, enc_v, _ = self._physics_tick(v_target, w_des, max_dv=0.65 * dt, max_dw=8.0 * dt)
            hits += hit
            s_enc += max(0.0, enc_v) * dt

            bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
            if math.hypot(bx - gx, by - gy) < 0.10:
                break

            if rem <= 0.012 and abs(enc_v) < 0.04:
                settled += 1
                if settled >= 6:
                    break
            else:
                settled = 0

        return hits

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
            else (65 if N <= 5 else (90 if N == 6 else 120))
        )
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_act = 0
        self._est_yaw = 0.0
        self._cardinal_yaw = 0.0
        self._last_slip = 0.0
        self.visited_cells = {(0, 0)}

        # Settle & auto-calibrate centered wall ToF distances from start corner (0,0)
        fb_samples, lr_samples = [], []
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            fb_samples.append(self._read_rf("rf_back"))
            lr_samples.append(min(self._read_rf("rf_left"), self._read_rf("rf_right")))

        self._center_fb = float(np.clip(np.mean(fb_samples), 0.095, 0.175))
        self._center_lr = float(np.clip(np.mean(lr_samples), 0.095, 0.175))

        return self._get_obs(), {
            "track_seed": track_seed,
            "terrain": self.current_terrain,
            "grid_size": N,
            "total_length": self.track["total_length"],
        }

    def step(self, action):
        act = int(np.asarray(action).item())
        self.steps += 1

        step_hits = self._exec_turn(MACRO_TURN_DELTAS[act])
        step_hits += self._exec_forward_one_cell()

        obs = self._get_obs()
        gx, gy = self.track["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        at_goal = math.hypot(bx - gx, by - gy) < 0.22

        cs = self.track["cell_size"]
        N = self.track["grid_size"]
        cell = (
            int(np.clip(round(bx / cs), 0, N - 1)),
            int(np.clip(round(by / cs), 0, N - 1)),
        )
        new_cell = cell not in self.visited_cells
        self.visited_cells.add(cell)

        reward = (1.0 if new_cell else -0.05) - 0.1 * step_hits
        crashed = self.wall_hits >= 25
        if crashed:
            reward -= 10.0
        if at_goal:
            reward += 30.0 + (10.0 if self.wall_hits == 0 else 0.0)

        self._prev_act = act
        terminated = at_goal or crashed
        truncated = (not terminated) and (self.steps >= self.max_steps)

        info = {
            "is_success": at_goal,
            "crashed": crashed,
            "terrain": self.current_terrain,
            "wall_hits": self.wall_hits,
            "visited_cells": len(self.visited_cells),
        }
        return obs, float(reward), terminated, truncated, info
