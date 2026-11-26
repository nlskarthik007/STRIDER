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
        "fric": [0.25, 0.001, 0.0001],
        "solref": [0.020, 1.00],
        "solimp": [0.90, 0.95, 0.001, 0.5, 2.0],
        "rgba": [0.76, 0.88, 0.96, 1.0],
        "drag": 0.0,
    },
}


class RealWorldCurveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        terrain: str = "random",
        corridor_width: float = 0.44,
        max_steps: int = 340,
        base_seed: int = 0,
        noise_enabled: bool = True,
        render_realtime: bool = False,
    ):
        super().__init__()
        self.buggy_xml_path = buggy_xml_path
        self.terrain_mode = terrain
        self.corridor_width = corridor_width
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.noise_enabled = noise_enabled
        self.render_realtime = render_realtime

        self.wheel_radius = 0.03
        self.track_width = 0.12
        self.max_wheel_vel = 35.0
        self.max_range = 1.20
        self.ticks_per_step = 10

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._prev_act = np.zeros(2, dtype=np.float32)

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

    def apply_terrain(self, terrain: str):
        cfg = TERRAIN_PRESETS[terrain]
        self.current_terrain = terrain
        self.sand_drag = cfg["drag"]

        fric = np.array(cfg["fric"], dtype=np.float64)
        if self.terrain_mode == "random":
            fric[0] *= float(self.rng.uniform(0.88, 1.12))
            self.sand_drag *= float(self.rng.uniform(0.85, 1.15))

        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "floor" in gname or "ground" in gname or gid == 0:
                self.model.geom_friction[gid] = fric
                self.model.geom_solref[gid] = cfg["solref"]
                self.model.geom_solimp[gid] = cfg["solimp"]
                self.model.geom_rgba[gid] = cfg["rgba"]

    def _read_rf(self, key: str) -> float:
        adr = self._sensor_adr.get(key, -1)
        if adr < 0:
            return self.max_range
        val = float(self.data.sensordata[adr])
        if val < 0.0 or val > self.max_range:
            val = self.max_range
        if self.noise_enabled:
            val = float(np.clip(val + self.rng.normal(0.0, 0.008), 0.01, self.max_range))
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
            wl += float(self.rng.normal(0.0, 0.10))
            wr += float(self.rng.normal(0.0, 0.10))
        return wl, wr

    def _read_gyro_z(self) -> float:
        adr = self._sensor_adr.get("gyro", -1)
        if adr < 0:
            return 0.0
        gz = float(self.data.sensordata[adr + 2])
        if self.noise_enabled:
            gz += float(self.rng.normal(0.0, 0.02))
        return gz

    def _track_metrics(self):
        pts = self.track["centerline"]
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
        return float(idx * 0.14), float(dists[idx]), float(v_along)

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

        lat_bal = (rf_l - rf_r) / self.max_range
        diag_bal = (rf_fl - rf_fr) / self.max_range

        gx, gy = self.track["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        dist_g = math.hypot(gx - bx, gy - by)

        obs = np.array(
            [
                rf_f / self.max_range,
                rf_b / self.max_range,
                rf_l / self.max_range,
                rf_r / self.max_range,
                rf_fl / self.max_range,
                rf_fr / self.max_range,
                np.clip(enc_v / 0.60, -1.0, 1.0),
                np.clip(gyro_z / 4.0, -1.0, 1.0),
                np.clip(yaw_slip / 4.0, -1.0, 1.0),
                np.clip(lat_bal * 2.5, -1.0, 1.0),
                np.clip(diag_bal * 2.5, -1.0, 1.0),
                min(dist_g / 6.0, 1.0),
                float(self._prev_act[0]),
                float(self._prev_act[1]),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        track_seed = int(seed) if seed is not None else int(self.rng.integers(0, 100000))

        self.model, self.data, self.track = build_curved_track_scene(
            self.buggy_xml_path,
            seed=track_seed,
            corridor_width=self.corridor_width,
        )
        self._cache_ids()

        if self.terrain_mode == "random":
            t_choice = str(
                self.rng.choice(["sand", "pavement", "gravel", "ice"], p=[0.40, 0.20, 0.20, 0.20])
            )
        else:
            t_choice = self.terrain_mode
        self.apply_terrain(t_choice)

        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._prev_act = np.zeros(2, dtype=np.float32)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {"track_seed": track_seed, "terrain": self.current_terrain}

    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1

        v_target = 0.30 + 0.25 * float(act[0])  # [0.05, 0.55] m/s
        w_target = 3.00 * float(act[1])

        dt = self.model.opt.timestep
        max_dv = 1.6 * dt
        max_dw = 12.0 * dt
        step_hits = 0

        for _ in range(self.ticks_per_step):
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
        progress, lat_err, v_along = self._track_metrics()

        gx, gy = self.track["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        at_goal = math.hypot(bx - gx, by - gy) < 0.20

        center_factor = max(0.0, 1.0 - lat_err / (0.5 * self.corridor_width))
        smooth_pen = float(np.sum((act - self._prev_act) ** 2))
        slip_pen = abs(float(obs[8]))

        # Speed-coupled tangential reward: zero payout for crawling/stalling
        reward = (
            0.70 * v_along * (0.35 + 0.65 * center_factor)
            - 0.03
            - 0.015 * smooth_pen
            - 0.03 * slip_pen
        )

        if step_hits > 0:
            self._consec_hits += 1
            reward -= 0.4
        else:
            self._consec_hits = 0

        crashed = self._consec_hits >= 5
        if crashed:
            reward -= 5.0
        if at_goal:
            reward += 25.0

        self._prev_act = act
        terminated = at_goal or crashed
        truncated = (not terminated) and (self.steps >= self.max_steps)

        info = {
            "is_success": at_goal,
            "crashed": crashed,
            "terrain": self.current_terrain,
            "wall_hits": self.wall_hits,
            "progress_m": progress,
        }
        return obs, float(reward), terminated, truncated, info