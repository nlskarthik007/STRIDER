import math
from collections import defaultdict
import numpy as np
import mujoco


class PlanarLidarSensorSuite:
    """
    28-D Real-World Sensor & Local Odometry Memory Suite for RL_buggy_2.0:
      [0:16]  16-ray 360-deg planar LIDAR (normalized to max_range, noisy)
      [16:21] enc_v, gyro_z, yaw_slip, prev_v_norm, prev_w_norm
      [21:24] ego-frame Euclidean goal (euc_dist_norm, cos_rel, sin_rel)
      [24:28] 4-sector local visited density [Front, Left, Back, Right] via odometry
    """

    def __init__(
        self,
        num_rays: int = 16,
        max_range: float = 2.0,
        wheel_radius: float = 0.032,
        track_width: float = 0.135,
        grid_res: float = 0.25,
        noise_enabled: bool = True,
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.grid_res = grid_res
        self.noise_enabled = noise_enabled

        self.lidar_adr = []
        self.gyro_adr = -1
        self.vel_l_adr = -1
        self.vel_r_adr = -1

        # Onboard dead-reckoning state (wheel encoders + Z-gyro only)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.visit_counts = defaultdict(float)

    def bind(self, model: mujoco.MjModel, prefix: str = "bot/"):
        self.lidar_adr = []
        for i in range(self.num_rays):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}lidar_{i:02d}")
            assert sid >= 0, f"Missing LIDAR sensor {prefix}lidar_{i:02d}"
            self.lidar_adr.append(int(model.sensor_adr[sid]))

        def _find_sensor(candidates):
            for name in candidates:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}{name}")
                if sid >= 0:
                    return int(model.sensor_adr[sid])
            return -1

        self.gyro_adr = _find_sensor(["gyro", "imu_gyro", "chassis_gyro"])
        self.vel_l_adr = _find_sensor(["left_wheel_vel", "wheel_l_vel", "enc_l"])
        self.vel_r_adr = _find_sensor(["right_wheel_vel", "wheel_r_vel", "enc_r"])

    def reset_odom(self, init_x: float = 0.0, init_y: float = 0.0, init_yaw: float = 0.0):
        self.odom_x = float(init_x)
        self.odom_y = float(init_y)
        self.odom_yaw = float(init_yaw)
        self.visit_counts = defaultdict(float)
        self._mark_visit(self.odom_x, self.odom_y, amount=1.0)

    def _mark_visit(self, x: float, y: float, amount: float = 0.15):
        gx = int(round(x / self.grid_res))
        gy = int(round(y / self.grid_res))
        self.visit_counts[(gx, gy)] = min(5.0, self.visit_counts[(gx, gy)] + amount)

    def read_lidar(self, data: mujoco.MjData, rng: np.random.Generator) -> np.ndarray:
        raw = np.empty(self.num_rays, dtype=np.float32)
        for i, adr in enumerate(self.lidar_adr):
            d = float(data.sensordata[adr])
            if d < 0.0 or d > self.max_range:
                d = self.max_range
            if self.noise_enabled:
                # Realistic ToF/LIDAR noise (0.8cm base + 1% range) + 1.5% dropout
                if rng.random() < 0.015:
                    d = self.max_range
                else:
                    d += float(rng.normal(0.0, 0.008 + 0.01 * d))
            raw[i] = float(np.clip(d, 0.0, self.max_range))
        return raw

    def read_gyro_z(self, data: mujoco.MjData, rng: np.random.Generator) -> float:
        if self.gyro_adr >= 0:
            gz = float(data.sensordata[self.gyro_adr + 2])
        else:
            gz = float(data.qvel[5])
        if self.noise_enabled:
            gz += float(rng.normal(0.0, 0.025))
        return gz

    def read_wheel_kinematics(self, data: mujoco.MjData, rng: np.random.Generator):
        if self.vel_l_adr >= 0 and self.vel_r_adr >= 0:
            wl = float(data.sensordata[self.vel_l_adr])
            wr = float(data.sensordata[self.vel_r_adr])
        else:
            # Fallback to joint velocities if explicit encoder sensors are unnamed
            wl = float(data.qvel[6]) if len(data.qvel) > 7 else 0.0
            wr = float(data.qvel[7]) if len(data.qvel) > 7 else 0.0
        if self.noise_enabled:
            wl += float(rng.normal(0.0, 0.15))
            wr += float(rng.normal(0.0, 0.15))
        enc_v = 0.5 * (wl + wr) * self.wheel_radius
        enc_w = (wr - wl) * self.wheel_radius / self.track_width
        return enc_v, enc_w

    def update_odom(self, enc_v: float, gyro_z: float, dt: float):
        self.odom_yaw += gyro_z * dt
        self.odom_x += enc_v * math.cos(self.odom_yaw) * dt
        self.odom_y += enc_v * math.sin(self.odom_yaw) * dt
        self._mark_visit(self.odom_x, self.odom_y, amount=0.10)

    def sector_visit_density(self) -> np.ndarray:
        """Sample visited heat 0.35m ahead in [Front, Left, Back, Right] in ego frame."""
        probe = 0.35
        densities = np.zeros(4, dtype=np.float32)
        for idx, rel_ang in enumerate((0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi)):
            ang = self.odom_yaw + rel_ang
            px = self.odom_x + probe * math.cos(ang)
            py = self.odom_y + probe * math.sin(ang)
            gx = int(round(px / self.grid_res))
            gy = int(round(py / self.grid_res))
            # Normalize to [-1, 1]: -1 = completely unvisited, +1 = heavily revisited
            heat = min(1.0, self.visit_counts[(gx, gy)] / 2.5)
            densities[idx] = float(2.0 * heat - 1.0)
        return densities

    def build_obs(
        self,
        data: mujoco.MjData,
        rng: np.random.Generator,
        goal_xy: tuple[float, float],
        max_diag: float,
        prev_v_norm: float,
        prev_w_norm: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        lidar_m = self.read_lidar(data, rng)
        lidar_norm = (np.log1p(lidar_m / 0.18) / math.log1p(self.max_range / 0.18)) * 2.0 - 1.0

        enc_v, enc_w = self.read_wheel_kinematics(data, rng)
        gyro_z = self.read_gyro_z(data, rng)
        yaw_slip = enc_w - gyro_z

        bx, by = float(data.qpos[0]), float(data.qpos[1])
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        dx, dy = goal_xy[0] - bx, goal_xy[1] - by
        euc_dist = math.hypot(dx, dy)
        rel_ang = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi

        if self.noise_enabled:
            euc_obs = max(0.0, euc_dist + float(rng.normal(0.0, 0.015)))
            rel_obs = rel_ang + float(rng.normal(0.0, 0.02))
        else:
            euc_obs, rel_obs = euc_dist, rel_ang

        euc_norm = float(np.clip((euc_obs / (max_diag + 1e-6)) * 2.0 - 1.0, -1.0, 1.0))
        sectors = self.sector_visit_density()

        obs = np.concatenate(
            [
                lidar_norm,
                np.array(
                    [
                        np.clip(enc_v / 0.60, -1.0, 1.0),
                        np.clip(gyro_z / 3.0, -1.0, 1.0),
                        np.clip(yaw_slip / 3.0, -1.0, 1.0),
                        float(prev_v_norm),
                        float(prev_w_norm),
                        euc_norm,
                        math.cos(rel_obs),
                        math.sin(rel_obs),
                    ],
                    dtype=np.float32,
                ),
                sectors,
            ]
        )
        return np.clip(obs, -1.0, 1.0), lidar_m, euc_dist
