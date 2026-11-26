import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from sim.curve_scene_builder import build_curved_track_scene


TERRAIN_PRESETS = {
    "pavement": {
        "fric": [1.00, 0.005, 0.0001],
        "solref": [0.020, 1.00],
        "solimp": [0.90, 0.95, 0.001, 0.5, 2.0],
        "rgba": [0.28, 0.30, 0.34, 1.0],
        "drag": 0.0,
    },
    "sand": {
        "fric": [0.48, 0.020, 0.0150],
        "solref": [0.045, 0.75],
        "solimp": [0.75, 0.92, 0.015, 0.5, 2.0],
        "rgba": [0.78, 0.66, 0.46, 1.0],
        "drag": 0.6,
    },
    "gravel": {
        "fric": [0.62, 0.025, 0.0120],
        "solref": [0.032, 0.85],
        "solimp": [0.82, 0.94, 0.008, 0.5, 2.0],
        "rgba": [0.52, 0.50, 0.47, 1.0],
        "drag": 0.35,
    },
    "ice": {
        "fric": [0.08, 0.001, 0.0001],
        "solref": [0.020, 1.00],
        "solimp": [0.90, 0.95, 0.001, 0.5, 2.0],
        "rgba": [0.76, 0.88, 0.96, 1.0],
        "drag": 0.0,
    },
    # Unseen zero-shot stress-test surface (never named during training)
    "mud": {
        "fric": [0.35, 0.030, 0.0200],
        "solref": [0.055, 0.70],
        "solimp": [0.70, 0.90, 0.020, 0.5, 2.0],
        "rgba": [0.36, 0.25, 0.16, 1.0],
        "drag": 0.85,
    },
}

# Smooth tangential primitives: (nominal_speed_mps, curvature_rad_per_m)
ARC_PRIMITIVES = (
    (0.52,  0.00),  # 0: straight fast
    (0.42,  1.15),  # 1: gentle left arc
    (0.42, -1.15),  # 2: gentle right arc
    (0.26,  2.50),  # 3: sharp left arc
    (0.26, -2.50),  # 4: sharp right arc
)


class RealWorldHierCurveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        terrain: str = "random",
        corridor_width=0.44,
        track_type: str = "random",
        max_steps: int = None,
        base_seed: int = 0,
        noise_enabled: bool = True,
        render_realtime: bool = False,
    ):
        super().__init__()
        self.buggy_xml_path = buggy_xml_path
        self.terrain_mode = terrain
        self.corridor_width = corridor_width  # float or (min_w, max_w)
        self.current_corridor_width = (
            float(corridor_width[1]) if isinstance(corridor_width, (tuple, list)) else float(corridor_width)
        )
        self.track_type = track_type
        self.max_steps = max_steps if max_steps is not None else (420 if track_type == "marathon" else 280)
        self.base_seed = base_seed
        self.noise_enabled = noise_enabled
        self.render_realtime = render_realtime

        self.wheel_radius = 0.03
        self.track_width = 0.12
        self.max_wheel_vel = 35.0
        self.max_range = 1.20
        self.ticks_per_step = 20  # 10 Hz high-level policy over 200 Hz inner PID

        self.action_space = spaces.Discrete(len(ARC_PRIMITIVES))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

        self.rng = np.random.default_rng(base_seed)
        self.viewer = None
        self.model = None
        self.data = None
        self.track = None

        self._sensor_adr = {}
        self._act_l = -1
        self._act_r = -1
        self._dof_l = -1
        self._dof_r = -1
        self._wall_geom_ids = set()
        self._bot_geom_ids = set()

        self.current_terrain = "sand"
        self.sand_drag = 0.0
        self._cycle_idx = 0
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_act = 0

    def _resolve_id(self, obj_type, name: str) -> int:
        for cand in (f"bot/{name}", name):
            idx = mujoco.mj_name2id(self.model, obj_type, cand)
            if idx != -1:
                return idx
        raise KeyError(f"MuJoCo object '{name}' not found")

    def _cache_ids(self):
        names = [
            "rf_front", "rf_back", "rf_left", "rf_right",
            "rf_fl", "rf_fr", "gyro", "enc_left_vel", "enc_right_vel",
        ]
        self._sensor_adr = {}
        for name in names:
            try:
                sid = self._resolve_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
                self._sensor_adr[name] = int(self.model.sensor_adr[sid])
            except KeyError:
                self._sensor_adr[name] = -1

        self._act_l = self._resolve_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_vel")
        self._act_r = self._resolve_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_vel")
        jnt_l = int(self.model.actuator_trnid[self._act_l, 0])
        jnt_r = int(self.model.actuator_trnid[self._act_r, 0])
        self._dof_l = int(self.model.jnt_dofadr[jnt_l])
        self._dof_r = int(self.model.jnt_dofadr[jnt_r])

        self._wall_geom_ids = set()
        self._bot_geom_ids = set()
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname.startswith("wall_"):
                self._wall_geom_ids.add(gid)
            elif gname.startswith("bot/"):
                self._bot_geom_ids.add(gid)

    def _apply_floor_params(self, fric, solref, solimp, rgba, drag: float):
        self.sand_drag = float(drag)
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "floor" in gname or "ground" in gname or gid == 0:
                self.model.geom_priority[gid] = 1
                self.model.geom_friction[gid] = fric
                self.model.geom_solref[gid] = solref
                self.model.geom_solimp[gid] = solimp
                self.model.geom_rgba[gid] = rgba

    def apply_terrain(self, terrain: str):
        if terrain == "continuous_dr":
            self.current_terrain = "domain_rand"
            mu = float(self.rng.uniform(0.07, 1.00))
            drag = float(self.rng.uniform(0.0, 0.95))
            softness = np.clip(drag / 0.95, 0.0, 1.0)
            fric = np.array([mu, 0.002 + 0.030 * softness, 0.0005 + 0.020 * softness], dtype=np.float64)
            solref = [0.020 + 0.038 * softness, 1.00 - 0.30 * softness]
            solimp = [0.90 - 0.20 * softness, 0.95 - 0.05 * softness, 0.001 + 0.020 * softness, 0.5, 2.0]
            rgba = [0.55 - 0.15 * softness, 0.50 - 0.18 * softness, 0.45 - 0.20 * softness, 1.0]
            self._apply_floor_params(fric, solref, solimp, rgba, drag)
            return

        cfg = TERRAIN_PRESETS[terrain]
        self.current_terrain = terrain
        fric = np.array(cfg["fric"], dtype=np.float64)
        drag = cfg["drag"]
        if self.terrain_mode in ("random", "domain_rand"):
            fric[0] *= float(self.rng.uniform(0.88, 1.12))
            drag *= float(self.rng.uniform(0.85, 1.15))
        self._apply_floor_params(fric, cfg["solref"], cfg["solimp"], cfg["rgba"], drag)

    def _read_rf(self, key: str) -> float:
        adr = self._sensor_adr.get(key, -1)
        if adr < 0:
            return self.max_range
        val = float(self.data.sensordata[adr])
        if val < 0.0 or val > self.max_range:
            val = self.max_range
        if self.noise_enabled:
            val = float(np.clip(val + self.rng.normal(0.0, 0.006), 0.01, self.max_range))
        return val

    def _read_wheel_speeds(self):
        adr_l = self._sensor_adr.get("enc_left_vel", -1)
        adr_r = self._sensor_adr.get("enc_right_vel", -1)
        if adr_l >= 0 and adr_r >= 0:
            wl = float(self.data.sensordata[adr_l])
            wr = float(self.data.sensordata[adr_r])
        else:
            wl = float(self.data.qvel[self._dof_l])
            wr = float(self.data.qvel[self._dof_r])
        if self.noise_enabled:
            wl += float(self.rng.normal(0.0, 0.08))
            wr += float(self.rng.normal(0.0, 0.08))
        return wl, wr

    def _read_gyro_z(self) -> float:
        adr = self._sensor_adr.get("gyro", -1)
        if adr < 0:
            return 0.0
        gz = float(self.data.sensordata[adr + 2])
        if self.noise_enabled:
            gz += float(self.rng.normal(0.0, 0.015))
        return gz

    def _track_metrics(self):
        pts = self.track["centerline"]
        cum_len = self.track["cum_len"]
        half_w = self.track["half_w"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
        idx = int(np.argmin(dists))
        p0 = pts[max(0, idx - 1)] if idx == len(pts) - 1 else pts[idx]
        p1 = pts[idx] if idx == len(pts) - 1 else pts[idx + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg = math.hypot(dx, dy) + 1e-8
        tx, ty = dx / seg, dy / seg

        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        v_along = vx * tx + vy * ty
        return float(cum_len[idx]), float(dists[idx]), float(v_along), float(half_w[idx])

    def _count_wall_contacts(self) -> int:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._wall_geom_ids and g2 in self._bot_geom_ids) or (
                g2 in self._wall_geom_ids and g1 in self._bot_geom_ids
            ):
                return 1
        return 0

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
        # Purely local corridor-width cue (invariant to 180 deg hairpins & track length)
        corridor_cue = np.clip((rf_l + rf_r) / 0.45, 0.0, 1.0)

        prev_v, prev_k = ARC_PRIMITIVES[self._prev_act]

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
                prev_k / 2.50,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        track_seed = int(seed) if seed is not None else int(self.rng.integers(0, 100000))

        if isinstance(self.corridor_width, (tuple, list)):
            self.current_corridor_width = float(self.rng.uniform(self.corridor_width[0], self.corridor_width[1]))
        else:
            self.current_corridor_width = float(self.corridor_width)

        self.model, self.data, self.track = build_curved_track_scene(
            self.buggy_xml_path,
            seed=track_seed,
            corridor_width=self.current_corridor_width,
            track_type=self.track_type,
        )
        self._cache_ids()

        if self.terrain_mode == "domain_rand":
            if self.rng.random() < 0.40:
                t_choice = "continuous_dr"
            else:
                t_choice = str(self.rng.choice(["sand", "gravel", "pavement", "ice"], p=[0.30, 0.20, 0.20, 0.30]))
        elif self.terrain_mode == "random":
            t_choice = str(
                self.rng.choice(["sand", "pavement", "gravel", "ice"], p=[0.35, 0.15, 0.15, 0.35])
            )
        elif self.terrain_mode == "cycle":
            terrains = ("sand", "gravel", "pavement", "ice")
            t_choice = terrains[self._cycle_idx % 4]
            self._cycle_idx += 1
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

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {
            "track_seed": track_seed,
            "terrain": self.current_terrain,
            "track_type": self.track_type,
            "corridor_width": self.current_corridor_width,
            "total_length": self.track["total_length"],
        }

    def step(self, action):
        act = int(np.asarray(action).item())
        self.steps += 1

        v_nom, kappa_nom = ARC_PRIMITIVES[act]
        dt = self.model.opt.timestep
        max_dv = 2.0 * dt
        max_dw = 14.0 * dt
        step_hits = 0

        for _ in range(self.ticks_per_step):
            rf_f = self._read_rf("rf_front")
            rf_l = self._read_rf("rf_left")
            rf_r = self._read_rf("rf_right")
            rf_fl = self._read_rf("rf_fl")
            rf_fr = self._read_rf("rf_fr")

            # Low-level ToF speed governor (balanced: never times out on 0.36 m, controlled on 0.33 m pinch)
            front_scale = float(np.clip((rf_f - 0.10) / 0.22, 0.30, 1.0))
            width_scale = float(np.clip(((rf_l + rf_r) - 0.15) / 0.12, 0.55, 1.0))
            diag_scale = float(np.clip((min(rf_fl, rf_fr) - 0.09) / 0.10, 0.55, 1.0))
            v_target = v_nom * min(front_scale, width_scale, diag_scale)
            w_des = v_target * kappa_nom

            # Smooth linear wall-buffer shield (inactive at 0.33 m pinch center, zero ice-jerk)
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

            # Imminent-contact micro-brake gives Gyro PI time to pivot the nose clear
            if min(rf_l, rf_r) < 0.048 or min(rf_fl, rf_fr) < 0.072:
                v_target *= 0.45

            w_des = float(np.clip(w_des + repulse, -3.8, 3.8))

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
        progress, lat_err, v_along, local_half_w = self._track_metrics()

        gx, gy = self.track["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        at_goal = math.hypot(bx - gx, by - gy) < 0.22

        center_factor = max(0.0, 1.0 - lat_err / max(local_half_w, 0.10))
        switch_pen = 0.02 if act != self._prev_act else 0.0

        reward = (
            1.40 * v_along * (0.35 + 0.65 * center_factor)
            - 0.04
            - switch_pen
        )

        if step_hits > 0:
            self._consec_hits += 1
            reward -= 1.2
        else:
            self._consec_hits = 0

        crashed = (self._consec_hits >= 3) or (self.wall_hits >= 25)
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
            "progress_m": progress,
            "total_length": self.track["total_length"],
        }
        return obs, float(reward), terminated, truncated, info