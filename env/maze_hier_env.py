import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from sim.scene_builder import build_maze_scene


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MazeHierEnv(gym.Env):
    """
    Hierarchical MuJoCo maze navigation environment with:
    - Friction-adaptive traction control & ToF-fused slip-resistant odometry
    - Arbitrary-angle turns (any degree) via gyro-locked PD control
    - Continuous R=0.15 m quarter-circle arc cornering + straightaway lookahead
    """

    metadata = {"render_modes": ["human"]}

    # Cardinal headings: 0=East (+x), 1=North (+y), 2=West (-x), 3=South (-y)
    HEADING_DC_DR = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    HEADING_YAW = [0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi]
    REL_DELTA_H = [0, 1, 3, 2]  # 0:F, 1:L (+90), 2:R (-90), 3:B (180)

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        rows: int = 5,
        cols: int = 5,
        cell_size: float = 0.30,
        pool_size: int = 40,
        base_seed: int = 0,
        max_moves: int = 80,
        noise_enabled: bool = False,
        render_realtime: bool = False,
        cruise_speed: float = 0.60,
        brake_dist: float = 0.10,
        accel: float = 1.9,
        friction_range: tuple = (1.0, 1.0),
        enable_arc_turns: bool = True,
    ):
        super().__init__()
        self.buggy_xml_path = buggy_xml_path
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.pool_size = max(1, pool_size)
        self.base_seed = base_seed
        self.max_moves = max_moves
        self.noise_enabled = noise_enabled
        self.render_realtime = render_realtime

        # Motion & surface parameters
        self.base_cruise_speed = cruise_speed
        self.base_brake_dist = brake_dist
        self.base_accel = accel
        self.friction_range = friction_range
        self.current_friction = 1.0
        self.enable_arc_turns = enable_arc_turns

        # Active traction-scaled limits (updated per episode in reset)
        self.cruise_speed = cruise_speed
        self.brake_dist = brake_dist
        self.accel = accel

        # Kinematics & sensor geometry (matches assets/my_buggy.xml)
        self.wheel_radius = 0.03
        self.track_width = 0.12
        self.max_wheel_vel = 35.0
        self.sensor_off = 0.065
        self.max_range = 1.20
        self.open_thresh = self.cell_size * 0.72
        self.front_center = 0.5 * self.cell_size - self.sensor_off
        self.side_center = 0.5 * self.cell_size - self.sensor_off

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        self.rng = np.random.default_rng(base_seed)
        self.viewer = None
        self.model = None
        self.data = None
        self.maze = None

        # Cached MuJoCo IDs
        self._sensor_adr = {}
        self._act_l = -1
        self._act_r = -1
        self._dof_l = -1
        self._dof_r = -1
        self._wall_geom_ids = set()
        self._bot_geom_ids = set()

        # Episode state (continuous target_yaw supports any-degree turns)
        self.odom_r = 0
        self.odom_c = 0
        self.odom_h = 0
        self.target_yaw = 0.0
        self.odom_yaw = 0.0
        self.goal_r = 0
        self.goal_c = 0
        self.visited_counts = {}
        self.moves = 0
        self.wall_hits = 0
        self.collision_moves = 0
        self._move_wall_hits = 0

    # ------------------------------------------------------------------
    # MuJoCo ID Resolution, Friction Setup & Low-Level I/O
    # ------------------------------------------------------------------
    def _resolve_id(self, obj_type, name: str) -> int:
        for candidate in (f"bot/{name}", name):
            idx = mujoco.mj_name2id(self.model, obj_type, candidate)
            if idx != -1:
                return idx
        raise KeyError(
            f"MuJoCo object '{name}' not found (checked 'bot/{name}' and '{name}')"
        )

    def _cache_ids(self):
        sensor_names = [
            "rf_front", "rf_back", "rf_left", "rf_right",
            "rf_fl", "rf_fr", "gyro", "enc_left_vel", "enc_right_vel",
        ]
        self._sensor_adr = {}
        for name in sensor_names:
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

    def _apply_surface_friction(self, mu: float):
        """Set floor sliding friction and scale traction envelope to prevent wheel slip."""
        self.current_friction = float(mu)
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "floor" in gname or "ground" in gname or gid == 0:
                self.model.geom_friction[gid, 0] = self.current_friction

        # Scale acceleration & braking envelope smoothly with available surface grip
        grip_scale = float(np.clip(self.current_friction, 0.25, 1.0))
        self.accel = self.base_accel * (grip_scale ** 0.75)
        self.cruise_speed = self.base_cruise_speed * (0.65 + 0.35 * grip_scale)
        self.brake_dist = float(
            np.clip(self.base_brake_dist / max(grip_scale, 0.45), 0.09, 0.16)
        )

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

    def _read_rangefinders(self) -> dict:
        return {
            "front": self._read_rf("rf_front"),
            "back": self._read_rf("rf_back"),
            "left": self._read_rf("rf_left"),
            "right": self._read_rf("rf_right"),
            "fl": self._read_rf("rf_fl"),
            "fr": self._read_rf("rf_fr"),
        }

    def _read_gyro_z(self) -> float:
        adr = self._sensor_adr.get("gyro", -1)
        if adr < 0:
            return 0.0
        gz = float(self.data.sensordata[adr + 2])
        if self.noise_enabled:
            gz += float(self.rng.normal(0.0, 0.015))
        return gz

    def _read_forward_speed(self) -> float:
        adr_l = self._sensor_adr.get("enc_left_vel", -1)
        adr_r = self._sensor_adr.get("enc_right_vel", -1)
        if adr_l >= 0 and adr_r >= 0:
            wl = float(self.data.sensordata[adr_l])
            wr = float(self.data.sensordata[adr_r])
        elif self._dof_l >= 0 and self._dof_r >= 0:
            wl = float(self.data.qvel[self._dof_l])
            wr = float(self.data.qvel[self._dof_r])
        else:
            return 0.0
        if self.noise_enabled:
            wl += float(self.rng.normal(0.0, 0.08))
            wr += float(self.rng.normal(0.0, 0.08))
        return 0.5 * (wl + wr) * self.wheel_radius

    def _fused_step_distance(self, prev_rf: dict, cur_rf: dict, dt: float) -> float:
        """
        Slip-resistant longitudinal step estimator:
        Fuses wheel encoder velocity with front/back ToF range deltas so wheel spin
        or brake sliding on low-friction surfaces cannot corrupt distance tracking.
        """
        enc_ds = self._read_forward_speed() * dt
        tof_ds_list = []

        if prev_rf["front"] < 1.05 and cur_rf["front"] < 1.05:
            df = prev_rf["front"] - cur_rf["front"]
            if abs(df - enc_ds) < 0.015:  # reject sudden wall-opening discontinuities
                tof_ds_list.append(df)

        if prev_rf["back"] < 1.05 and cur_rf["back"] < 1.05:
            db = cur_rf["back"] - prev_rf["back"]
            if abs(db - enc_ds) < 0.015:
                tof_ds_list.append(db)

        if tof_ds_list:
            tof_ds = float(np.mean(tof_ds_list))
            # Trust ToF heavily on slippery surfaces where wheels slip
            w_tof = 0.75 if self.current_friction < 0.65 else 0.45
            return (1.0 - w_tof) * enc_ds + w_tof * tof_ds
        return enc_ds

    def _apply_wheel_vel(self, w_l: float, w_r: float):
        self.data.ctrl[self._act_l] = float(
            np.clip(w_l, -self.max_wheel_vel, self.max_wheel_vel)
        )
        self.data.ctrl[self._act_r] = float(
            np.clip(w_r, -self.max_wheel_vel, self.max_wheel_vel)
        )

    def _set_vw(self, v: float, w: float):
        w_l = (v - 0.5 * w * self.track_width) / self.wheel_radius
        w_r = (v + 0.5 * w * self.track_width) / self.wheel_radius
        self._apply_wheel_vel(w_l, w_r)

    def _count_wall_contacts(self) -> int:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._wall_geom_ids and g2 in self._bot_geom_ids) or (
                g2 in self._wall_geom_ids and g1 in self._bot_geom_ids
            ):
                return 1
        return 0

    def _sim_tick(self):
        if self.viewer is not None and hasattr(self.viewer, "lock"):
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data)

        self.odom_yaw = _wrap_angle(
            self.odom_yaw + self._read_gyro_z() * self.model.opt.timestep
        )
        hit = self._count_wall_contacts()
        if hit:
            self.wall_hits += 1
            self._move_wall_hits += 1

        if self.viewer is not None:
            self.viewer.sync()
            if self.render_realtime:
                time.sleep(self.model.opt.timestep)

    # ------------------------------------------------------------------
    # Odometry, Goal & Observation (20-Dim)
    # ------------------------------------------------------------------
    def _advance_odom_cell(self):
        dc, dr = self.HEADING_DC_DR[self.odom_h]
        self.odom_c += dc
        self.odom_r += dr
        key = (self.odom_r, self.odom_c)
        self.visited_counts[key] = self.visited_counts.get(key, 0) + 1

    def _check_goal(self) -> bool:
        gx, gy = self.maze["goal_xy"]
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        return math.hypot(bx - gx, by - gy) < (self.cell_size * 0.38)

    def _get_obs(self) -> np.ndarray:
        rf = self._read_rangefinders()
        cardinal = [rf["front"], rf["left"], rf["right"], rf["back"]]

        open_dirs = [1.0 if d > self.open_thresh else 0.0 for d in cardinal]
        prox_norm = [min(d, self.max_range) / self.max_range for d in cardinal]

        adj_visited = []
        adj_counts = []
        for rel_a in range(4):
            h_adj = (self.odom_h + self.REL_DELTA_H[rel_a]) % 4
            dc, dr = self.HEADING_DC_DR[h_adj]
            cnt = self.visited_counts.get((self.odom_r + dr, self.odom_c + dc), 0)
            adj_visited.append(1.0 if cnt > 0 else 0.0)
            adj_counts.append(min(cnt, 4) / 4.0)

        dx_w = (self.goal_c - self.odom_c) * self.cell_size
        dy_w = (self.goal_r - self.odom_r) * self.cell_size
        dist_g = math.hypot(dx_w, dy_w)
        if dist_g > 1e-6:
            ux, uy = dx_w / dist_g, dy_w / dist_g
        else:
            ux, uy = 0.0, 0.0
        cy, sy = math.cos(self.odom_yaw), math.sin(self.odom_yaw)
        goal_local_x = ux * cy + uy * sy
        goal_local_y = -ux * sy + uy * cy

        obs = np.array(
            open_dirs
            + prox_norm
            + adj_visited
            + adj_counts
            + [goal_local_x, goal_local_y, sy, cy],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Arbitrary-Angle Turns, Curved Arcs & Low-Level Motion Primitives
    # ------------------------------------------------------------------
    def _compute_corridor_steering(self, rf: dict) -> float:
        yaw_err = _wrap_angle(self.target_yaw - self.odom_yaw)
        gyro_z = self._read_gyro_z()

        w_cmd = 5.0 * yaw_err - 0.24 * gyro_z

        side_thresh = self.cell_size * 0.65
        rf_l, rf_r = rf["left"], rf["right"]
        if rf_l < side_thresh and rf_r < side_thresh:
            w_cmd += 3.4 * (rf_l - rf_r)
        elif rf_l < side_thresh:
            w_cmd += 2.6 * (rf_l - self.side_center)
        elif rf_r < side_thresh:
            w_cmd += 2.6 * (self.side_center - rf_r)

        diag_safe = 0.092
        if rf["fl"] < diag_safe:
            w_cmd -= 4.0 * (diag_safe - rf["fl"])
        if rf["fr"] < diag_safe:
            w_cmd += 4.0 * (diag_safe - rf["fr"])

        return float(np.clip(w_cmd, -4.5, 4.5))

    def _settle_stop(self, max_ticks: int = 8):
        for _ in range(max_ticks):
            self._apply_wheel_vel(0.0, 0.0)
            self._sim_tick()
            if abs(self._read_forward_speed()) < 0.025 and abs(self._read_gyro_z()) < 0.15:
                break

    def turn_to_yaw(self, target_yaw: float, max_ticks: int = 260):
        """Rotate in place to ANY target angle (in radians) with gyro-damped PD control."""
        self.target_yaw = _wrap_angle(float(target_yaw))
        max_w = 7.5 * float(np.clip(self.current_friction, 0.35, 1.0))
        kp = 7.5 * float(np.clip(self.current_friction, 0.45, 1.0))
        kd = 0.34

        settled = 0
        for _ in range(max_ticks):
            yaw_err = _wrap_angle(self.target_yaw - self.odom_yaw)
            gyro_z = self._read_gyro_z()

            if abs(yaw_err) < 0.016 and abs(gyro_z) < 0.18:
                settled += 1
                self._set_vw(0.0, 0.0)
                self._sim_tick()
                if settled >= 2:
                    break
            else:
                settled = 0
                w_cmd = float(np.clip(kp * yaw_err - kd * gyro_z, -max_w, max_w))
                if abs(yaw_err) > 0.016 and abs(w_cmd) < 0.50:
                    w_cmd = math.copysign(0.50, yaw_err)
                self._set_vw(0.0, w_cmd)
                self._sim_tick()

        self._apply_wheel_vel(0.0, 0.0)

    def turn_by_degrees(self, degrees: float, max_ticks: int = 260):
        """Rotate relative to current target heading by ANY angle in degrees (+CCW, -CW)."""
        new_yaw = _wrap_angle(self.target_yaw + math.radians(float(degrees)))
        self.turn_to_yaw(new_yaw, max_ticks=max_ticks)

    def turn_in_place(self, rel_action: int, max_ticks: int = 260):
        """Phase A: Cardinal 90°/180° turn built on top of arbitrary-angle turn_to_yaw."""
        delta_h = self.REL_DELTA_H[int(rel_action)]
        if delta_h == 0:
            return
        self.odom_h = (self.odom_h + delta_h) % 4
        self.turn_to_yaw(self.HEADING_YAW[self.odom_h], max_ticks=max_ticks)

    def drive_arc_turn(self, turn_action: int, max_ticks: int = 220) -> bool:
        """
        Continuous quarter-circle arc cornering (R = 0.5 * cell_size = 0.15 m).
        Transitions smoothly from cell entrance (dist=0.15 m) onto the perpendicular
        corridor centerline without braking to zero.
        """
        assert turn_action in (1, 2), "Arc turn only supports Left (1) or Right (2)"
        turn_sign = 1.0 if turn_action == 1 else -1.0
        R = 0.5 * self.cell_size  # 0.15 m radius aligns entry & exit midpoints
        arc_len = 0.5 * math.pi * R

        start_yaw = self.target_yaw
        self.odom_h = (self.odom_h + self.REL_DELTA_H[turn_action]) % 4
        end_yaw = self.HEADING_YAW[self.odom_h]
        self.target_yaw = end_yaw

        v_arc = min(0.30, self.cruise_speed * 0.52)
        s = 0.0
        dt = self.model.opt.timestep
        prev_rf = self._read_rangefinders()

        for _ in range(max_ticks):
            rf = self._read_rangefinders()
            # Abort arc safely into straight brake if inner/outer diagonal clips
            if rf["fl"] < 0.072 or rf["fr"] < 0.072:
                self._settle_stop()
                self.turn_to_yaw(end_yaw)
                return False

            prog = float(np.clip(s / arc_len, 0.0, 1.0))
            ref_yaw = _wrap_angle(start_yaw + turn_sign * 0.5 * math.pi * prog)
            yaw_err = _wrap_angle(ref_yaw - self.odom_yaw)
            gyro_z = self._read_gyro_z()

            # Feedforward curvature w = v/R + gyro-damped trajectory tracking
            w_ff = turn_sign * (v_arc / R)
            w_cmd = float(np.clip(w_ff + 6.5 * yaw_err - 0.25 * gyro_z, -4.5, 4.5))

            self._set_vw(v_arc, w_cmd)
            self._sim_tick()

            ds = max(0.0, self._read_forward_speed() * dt)
            s += ds
            prev_rf = rf

            if prog >= 0.96 and abs(_wrap_angle(end_yaw - self.odom_yaw)) < 0.05:
                break

        # Final micro-trim to exact cardinal heading if friction caused slight understeer
        if abs(_wrap_angle(end_yaw - self.odom_yaw)) > 0.03:
            self.turn_to_yaw(end_yaw, max_ticks=60)
        return True

    def _execute_move(
        self,
        rel_action: int,
        cruise_speed: float = None,
        brake_dist: float = None,
        accel: float = None,
        max_ticks: int = 300,
    ):
        """Standard stop-and-go cell move used during RL training and step()."""
        cruise_speed = self.cruise_speed if cruise_speed is None else cruise_speed
        brake_dist = self.brake_dist if brake_dist is None else brake_dist
        accel = self.accel if accel is None else accel

        self.turn_in_place(rel_action)

        dist = 0.0
        dt = self.model.opt.timestep
        v_cmd = 0.12
        a_brake = 0.85 * (cruise_speed ** 2) / (2.0 * brake_dist)
        prev_rf = self._read_rangefinders()

        for _ in range(max_ticks):
            rf = self._read_rangefinders()
            rf_f = rf["front"]
            remaining = max(0.0, self.cell_size - dist)
            wall_rem = max(0.0, rf_f - self.front_center)
            effective_rem = min(remaining, wall_rem)

            if effective_rem <= brake_dist:
                v_target = max(0.07, math.sqrt(2.0 * a_brake * effective_rem))
                v_cmd = min(v_cmd, v_target)
            else:
                v_cmd = min(cruise_speed, v_cmd + accel * dt)

            w_cmd = self._compute_corridor_steering(rf)
            self._set_vw(v_cmd, w_cmd)
            self._sim_tick()

            dist += self._fused_step_distance(prev_rf, rf, dt)
            prev_rf = rf

            if dist > 0.16 and rf_f < self.front_center + 0.012:
                break
            if dist >= self.cell_size:
                break

        self._settle_stop()
        self._advance_odom_cell()

    def drive_cell_lookahead(
        self,
        peek_fn,
        carry: float = 0.0,
        brake_dist: float = None,
        cruise_speed: float = None,
        accel: float = None,
        max_ticks: int = 300,
    ) -> dict:
        """
        Phase B with slip-resistant ToF-fused odometry, non-stop straightaway chaining,
        and continuous R=0.15 m curved arc turns at clear 90° corners.
        """
        cruise_speed = self.cruise_speed if cruise_speed is None else cruise_speed
        brake_dist = self.brake_dist if brake_dist is None else brake_dist
        accel = self.accel if accel is None else accel

        dist = carry
        peeked = False
        next_action = None
        stop_at_end = True
        arc_corner = False
        hit_wall = False
        at_goal = False

        # Peek slightly earlier (0.16 m) when arc turns are enabled so R=0.15 m aligns cleanly
        peek_dist = 0.16 if self.enable_arc_turns else (self.cell_size - brake_dist)
        dt = self.model.opt.timestep
        v_cmd = max(0.12, self._read_forward_speed())
        a_brake = 0.85 * (cruise_speed ** 2) / (2.0 * brake_dist)
        prev_rf = self._read_rangefinders()

        for _ in range(max_ticks):
            rf = self._read_rangefinders()
            rf_f = rf["front"]

            if not peeked and dist >= peek_dist:
                peeked = True
                self._advance_odom_cell()
                at_goal = self._check_goal()
                obs = self._get_obs()

                if at_goal:
                    stop_at_end = True
                else:
                    next_action = int(peek_fn(obs))
                    front_open = (obs[0] > 0.5) and (rf_f > self.cell_size * 0.85)
                    side_clear = (
                        (next_action == 1 and obs[1] > 0.5 and rf["fl"] > 0.10)
                        or (next_action == 2 and obs[2] > 0.5 and rf["fr"] > 0.10)
                    )
                    if next_action == 0 and front_open:
                        stop_at_end = False
                    elif (
                        self.enable_arc_turns
                        and next_action in (1, 2)
                        and side_clear
                        and self.current_friction >= 0.45
                    ):
                        stop_at_end = False
                        arc_corner = True
                    else:
                        stop_at_end = True

            remaining = max(0.0, self.cell_size - dist)
            wall_rem = max(0.0, rf_f - self.front_center)

            if arc_corner:
                # Decelerate smoothly to arc entry speed (0.30 m/s) at junction entrance
                v_cmd = max(0.28, v_cmd - 1.8 * dt)
            else:
                effective_rem = min(remaining, wall_rem) if stop_at_end else wall_rem
                if effective_rem <= brake_dist:
                    v_target = max(0.07, math.sqrt(2.0 * a_brake * effective_rem))
                    v_cmd = min(v_cmd, v_target)
                else:
                    v_cmd = min(cruise_speed, v_cmd + accel * dt)

            w_cmd = self._compute_corridor_steering(rf)
            self._set_vw(v_cmd, w_cmd)
            self._sim_tick()

            dist += self._fused_step_distance(prev_rf, rf, dt)
            prev_rf = rf

            if arc_corner and dist >= 0.16:
                break
            if stop_at_end and dist > 0.16 and rf_f < self.front_center + 0.012:
                break
            if rf_f < self.front_center - 0.015:
                hit_wall = True
                stop_at_end = True
                break
            if dist >= self.cell_size:
                break

        if not peeked and dist > 0.15:
            self._advance_odom_cell()

        if arc_corner and not hit_wall and not at_goal:
            arc_ok = self.drive_arc_turn(next_action)
            # Exiting R=0.15 m arc places buggy 0.15 m past junction center -> carry=0.15
            return {
                "next_action": 0,  # Already turned onto new heading; drive straight into next cell
                "stopped": not arc_ok,
                "carry": 0.15 if arc_ok else 0.0,
                "hit_wall": hit_wall,
                "at_goal": self._check_goal(),
            }

        if stop_at_end:
            self._settle_stop()
            new_carry = 0.0
        else:
            new_carry = max(0.0, dist - self.cell_size)

        return {
            "next_action": next_action,
            "stopped": stop_at_end,
            "carry": new_carry,
            "hit_wall": hit_wall,
            "at_goal": at_goal or self._check_goal(),
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            maze_seed = int(seed)
        else:
            maze_seed = int(self.base_seed + self.rng.integers(0, self.pool_size))

        self.model, self.data, self.maze = build_maze_scene(
            self.buggy_xml_path,
            rows=self.rows,
            cols=self.cols,
            cell_size=self.cell_size,
            seed=maze_seed,
        )
        self._cache_ids()

        mu = float(self.rng.uniform(self.friction_range[0], self.friction_range[1]))
        self._apply_surface_friction(mu)

        sx, sy = self.maze["start_xy"]
        gx, gy = self.maze["goal_xy"]
        self.odom_c = int(sx // self.cell_size)
        self.odom_r = int(sy // self.cell_size)
        self.goal_c = int(gx // self.cell_size)
        self.goal_r = int(gy // self.cell_size)
        self.odom_h = 0
        self.target_yaw = 0.0
        self.odom_yaw = 0.0
        self.visited_counts = {(self.odom_r, self.odom_c): 1}
        self.moves = 0
        self.wall_hits = 0
        self.collision_moves = 0
        self._move_wall_hits = 0

        for _ in range(10):
            self._apply_wheel_vel(0.0, 0.0)
            self._sim_tick()
        self.wall_hits = 0
        self._move_wall_hits = 0

        return self._get_obs(), {"maze_seed": maze_seed, "friction": self.current_friction}

    def step(self, action):
        action = int(action)
        self.moves += 1
        self._move_wall_hits = 0

        obs_before = self._get_obs()
        old_dist = abs(self.goal_r - self.odom_r) + abs(self.goal_c - self.odom_c)

        if obs_before[action] < 0.5:
            reward = -1.0
            terminated = False
            truncated = self.moves >= self.max_moves
            info = {
                "is_success": False,
                "blocked": True,
                "wall_hits": self.wall_hits,
                "collision_moves": self.collision_moves,
            }
            return obs_before, reward, terminated, truncated, info

        self._execute_move(action)
        if self._move_wall_hits > 0:
            self.collision_moves += 1

        new_dist = abs(self.goal_r - self.odom_r) + abs(self.goal_c - self.odom_c)
        visits = self.visited_counts.get((self.odom_r, self.odom_c), 1)
        at_goal = self._check_goal()

        reward = -0.05 + 0.15 * (old_dist - new_dist)
        if visits == 1:
            reward += 0.25
        else:
            reward -= 0.10 * min(visits - 1, 3)
        if self._move_wall_hits > 0:
            reward -= 0.5
        if at_goal:
            reward += 10.0

        terminated = at_goal
        truncated = (not terminated) and (self.moves >= self.max_moves)
        info = {
            "is_success": at_goal,
            "blocked": False,
            "wall_hits": self.wall_hits,
            "collision_moves": self.collision_moves,
        }
        return self._get_obs(), float(reward), terminated, truncated, info