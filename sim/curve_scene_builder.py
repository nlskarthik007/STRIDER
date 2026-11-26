import math
import numpy as np
import mujoco


def generate_curved_centerline(
    seed: int,
    n_pts: int = 42,
    step: float = 0.14,
    track_type: str = "random",
):
    rng = np.random.default_rng(seed)
    pts = [(0.0, 0.0)]
    theta = 0.0
    sign = 1.0 if (seed % 2 == 0) else -1.0

    if track_type == "train_mix":
        mode = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.25, 0.20, 0.20]))
        if mode == 0:
            # High-curvature smooth clothoid S-bends
            curvature, target_curv = 0.0, 0.0
            for i in range(1, n_pts):
                if i <= 2:
                    target_curv = 0.0
                elif i % 5 == 3:
                    target_curv = float(rng.uniform(-1.90, 1.90))
                curvature = 0.50 * curvature + 0.50 * target_curv
                next_theta = theta + curvature * step
                if abs(next_theta) > 1.40:
                    target_curv = -abs(target_curv) if next_theta > 0 else abs(target_curv)
                    curvature = 0.50 * curvature + 0.50 * target_curv
                    next_theta = theta + curvature * step
                theta = next_theta
                pts.append((pts[-1][0] + step * math.cos(theta), pts[-1][1] + step * math.sin(theta)))

        elif mode == 1:
            # Randomized double switchback / hairpin (guaranteed >1.0 m leg separation)
            s_dir = 1.0 if rng.random() < 0.5 else -1.0
            lead = int(rng.integers(4, 7))
            turn_steps = int(rng.integers(12, 16))
            turn_angle = float(rng.uniform(2.65, 3.30))
            mid = int(rng.integers(5, 9))
            exit_len = int(rng.integers(5, 8))
            dtheta = turn_angle / turn_steps
            schedule = (
                [0.0] * lead
                + [s_dir * dtheta] * turn_steps
                + [0.0] * mid
                + [-s_dir * dtheta] * turn_steps
                + [0.0] * exit_len
            )
            for dth in schedule:
                theta += dth
                pts.append((pts[-1][0] + step * math.cos(theta), pts[-1][1] + step * math.sin(theta)))

        elif mode == 2:
            # Randomized rapid slalom / chicane
            s_dir = 1.0 if rng.random() < 0.5 else -1.0
            amp = float(rng.uniform(0.55, 0.76))
            wavelength = float(rng.uniform(2.10, 2.65))
            for i in range(1, 44):
                s = max(0.0, (i - 2) * step)
                env_ramp = min(1.0, s / 0.40)
                theta = s_dir * amp * env_ramp * math.sin(2.0 * math.pi * s / wavelength)
                pts.append((pts[-1][0] + step * math.cos(theta), pts[-1][1] + step * math.sin(theta)))

        else:
            # Clothoid curve paired with dynamic mid-track width pinch (handled in _compute_half_widths)
            curvature, target_curv = 0.0, 0.0
            for i in range(1, n_pts):
                if i <= 2:
                    target_curv = 0.0
                elif i % 6 == 3:
                    target_curv = float(rng.uniform(-1.65, 1.65))
                curvature = 0.55 * curvature + 0.45 * target_curv
                next_theta = theta + curvature * step
                if abs(next_theta) > 1.30:
                    target_curv = -abs(target_curv) if next_theta > 0 else abs(target_curv)
                    curvature = 0.55 * curvature + 0.45 * target_curv
                    next_theta = theta + curvature * step
                theta = next_theta
                pts.append((pts[-1][0] + step * math.cos(theta), pts[-1][1] + step * math.sin(theta)))

    elif track_type in ("random", "pinch", "marathon"):
        count = 76 if track_type == "marathon" else n_pts
        curvature = 0.0
        target_curv = 0.0
        for i in range(1, count):
            if i <= 2:
                target_curv = 0.0
            elif i % 6 == 3:
                target_curv = float(rng.uniform(-1.55, 1.55))

            curvature = 0.55 * curvature + 0.45 * target_curv
            next_theta = theta + curvature * step

            if abs(next_theta) > 1.25:
                target_curv = -abs(target_curv) if next_theta > 0 else abs(target_curv)
                curvature = 0.55 * curvature + 0.45 * target_curv
                next_theta = theta + curvature * step

            theta = next_theta
            x = pts[-1][0] + step * math.cos(theta)
            y = pts[-1][1] + step * math.sin(theta)
            pts.append((x, y))

    elif track_type == "chicane":
        count = 44
        for i in range(1, count):
            s = max(0.0, (i - 2) * step)
            env_ramp = min(1.0, s / 0.42)
            theta = sign * 0.65 * env_ramp * math.sin(2.0 * math.pi * s / 2.30)
            x = pts[-1][0] + step * math.cos(theta)
            y = pts[-1][1] + step * math.sin(theta)
            pts.append((x, y))

    elif track_type == "hairpin":
        dtheta = math.pi / 14.0
        schedule = (
            [0.0] * 6
            + [sign * dtheta] * 14
            + [0.0] * 7
            + [-sign * dtheta] * 14
            + [0.0] * 7
        )
        for dth in schedule:
            theta += dth
            x = pts[-1][0] + step * math.cos(theta)
            y = pts[-1][1] + step * math.sin(theta)
            pts.append((x, y))

    else:
        raise ValueError(f"Unknown track_type: {track_type}")

    return np.array(pts, dtype=np.float64)


def _compute_half_widths(
    n_pts: int,
    corridor_width: float,
    track_type: str,
    seed: int = 0,
) -> np.ndarray:
    if track_type == "pinch":
        t = np.linspace(0.0, 1.0, n_pts)
        widths = 0.48 - 0.15 * np.sin(np.pi * np.clip((t - 0.10) / 0.78, 0.0, 1.0)) ** 2
        widths = np.where(t > 0.88, 0.40, widths)
        return 0.5 * widths

    if track_type == "train_mix":
        wrng = np.random.default_rng(seed + 7777)
        if wrng.random() < 0.35:
            t = np.linspace(0.0, 1.0, n_pts)
            w_start = float(wrng.uniform(0.42, 0.48))
            w_min = float(wrng.uniform(0.32, 0.36))
            widths = w_start - (w_start - w_min) * np.sin(np.pi * np.clip((t - 0.10) / 0.78, 0.0, 1.0)) ** 2
            return 0.5 * widths

    return np.full(n_pts, 0.5 * corridor_width, dtype=np.float64)


def build_curved_track_scene(
    buggy_xml_path: str,
    seed: int = 0,
    corridor_width: float = 0.44,
    wall_height: float = 0.14,
    track_type: str = "random",
):
    pts = generate_curved_centerline(seed, track_type=track_type)
    n_pts = len(pts)
    half_w = _compute_half_widths(n_pts, corridor_width, track_type, seed=seed)

    tangents = np.zeros_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if n_pts > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])[:, None] + 1e-8
    tangents /= norms
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    pts_l = pts + half_w[:, None] * normals
    pts_r = pts - half_w[:, None] * normals

    wall_geoms = [
        f'<geom name="wall_start_cap" type="box" pos="-0.16 0.0 {0.5*wall_height:.4f}" '
        f'size="0.012 {half_w[0]:.4f} {0.5*wall_height:.4f}" rgba="0.22 0.28 0.36 1"/>'
    ]

    for i in range(n_pts - 1):
        for side, arr in (("L", pts_l), ("R", pts_r)):
            p0, p1 = arr[i], arr[i + 1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-4:
                continue
            ang = math.atan2(dy, dx)
            mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            wall_geoms.append(
                f'<geom name="wall_{side}_{i}" type="box" '
                f'pos="{mx:.4f} {my:.4f} {0.5*wall_height:.4f}" '
                f'euler="0 0 {ang:.4f}" '
                f'size="{0.56*seg_len:.4f} 0.012 {0.5*wall_height:.4f}" '
                f'rgba="0.22 0.28 0.36 1"/>'
            )

    seg_lens = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_length = float(cum_len[-1])

    gx, gy = pts[-1]
    goal_geom = (
        f'<geom name="goal_outer" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.002" '
        f'size="0.11 0.002" contype="0" conaffinity="0" rgba="0.06 0.72 0.50 0.85"/>'
        f'<geom name="goal_inner" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.003" '
        f'size="0.05 0.002" contype="0" conaffinity="0" rgba="1.0 1.0 1.0 0.95"/>'
    )

    scene_xml = f"""
    <mujoco model="curved_track_{track_type}">
      <option timestep="0.005" integrator="RK4"/>
      <visual>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/>
      </visual>
      <worldbody>
        <light pos="2.5 0 5.5" dir="0 0 -1"/>
        <geom name="terrain_floor" type="plane" size="35 35 0.1"
              rgba="0.78 0.66 0.46 1" friction="0.48 0.02 0.015"
              solref="0.045 0.75" solimp="0.75 0.92 0.015"/>
        {''.join(wall_geoms)}
        {goal_geom}
      </worldbody>
    </mujoco>
    """

    spec = mujoco.MjSpec.from_string(scene_xml)
    bot_spec = mujoco.MjSpec.from_file(buggy_xml_path)
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.035])
    if hasattr(spec, "attach"):
        spec.attach(bot_spec, prefix="bot/", frame=frame)
    else:
        frame.attach_body(bot_spec.worldbody.first_body(), "bot/", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, {
        "centerline": pts,
        "half_w": half_w,
        "cum_len": cum_len,
        "total_length": total_length,
        "goal_xy": (float(gx), float(gy)),
        "track_type": track_type,
    }