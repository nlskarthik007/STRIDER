import sys
import math
import time
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from env.lidar_sensor_suite import PlanarLidarSensorSuite


# Terrain physics table: (rgba, friction="slide torsion roll", quadratic_drag)
TERRAIN_SPECS = {
    "pavement": ("0.78 0.80 0.84 1", "1.00 0.005 0.0001", 0.00),
    "gravel":   ("0.60 0.46 0.34 1", "0.65 0.010 0.0200", 0.22),
    "sand":     ("0.90 0.78 0.36 1", "0.50 0.010 0.0450", 0.45),
    "ice":      ("0.66 0.88 0.98 1", "0.08 0.001 0.0001", 0.00),
    "mud":      ("0.36 0.25 0.18 1", "0.35 0.010 0.0600", 0.85),
}
TERRAIN_NAMES = tuple(TERRAIN_SPECS.keys())


def _sample_zone_centers(rng: np.random.Generator, xmin: float, xmax: float, ymin: float, ymax: float, mode: str):
    """Create 3-5 Voronoi terrain patches per episode so the buggy crosses multiple surfaces."""
    if mode in TERRAIN_SPECS:
        return [(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), mode)]
    k = int(rng.integers(3, 6))
    pool = ["pavement", "gravel", "sand", "ice", "mud"]
    chosen = list(rng.choice(pool, size=k, replace=False))
    centers = []
    for tname in chosen:
        cx = float(rng.uniform(xmin, xmax))
        cy = float(rng.uniform(ymin, ymax))
        centers.append((cx, cy, str(tname)))
    return centers


def _lookup_zone(x: float, y: float, centers):
    best_t = centers[0][2]
    best_d2 = float("inf")
    for cx, cy, tname in centers:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_t = tname
    return best_t


def _rasterize_box(grid: np.ndarray, ox: float, oy: float, res: float, cx: float, cy: float, hx: float, hy: float, yaw: float = 0.0):
    """Mark an oriented 2D box as occupied (1) in the occupancy grid."""
    H, W = grid.shape
    rad = math.hypot(hx, hy) + res
    ix0 = max(0, int(math.floor((cx - rad - ox) / res)))
    ix1 = min(W - 1, int(math.ceil((cx + rad - ox) / res)))
    iy0 = max(0, int(math.floor((cy - rad - oy) / res)))
    iy1 = min(H - 1, int(math.ceil((cy + rad - oy) / res)))
    if ix0 > ix1 or iy0 > iy1:
        return
    xs = ox + (np.arange(ix0, ix1 + 1) + 0.5) * res - cx
    ys = oy + (np.arange(iy0, iy1 + 1) + 0.5) * res - cy
    X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx = c * X - s * Y
    ly = s * X + c * Y
    mask = (np.abs(lx) <= (hx + 0.5 * res)) & (np.abs(ly) <= (hy + 0.5 * res))
    grid[iy0 : iy1 + 1, ix0 : ix1 + 1] |= mask.astype(np.uint8)


def _rasterize_cyl(grid: np.ndarray, ox: float, oy: float, res: float, cx: float, cy: float, r: float):
    H, W = grid.shape
    ix0 = max(0, int(math.floor((cx - r - res - ox) / res)))
    ix1 = min(W - 1, int(math.ceil((cx + r + res - ox) / res)))
    iy0 = max(0, int(math.floor((cy - r - res - oy) / res)))
    iy1 = min(H - 1, int(math.ceil((cy + r + res - oy) / res)))
    if ix0 > ix1 or iy0 > iy1:
        return
    xs = ox + (np.arange(ix0, ix1 + 1) + 0.5) * res - cx
    ys = oy + (np.arange(iy0, iy1 + 1) + 0.5) * res - cy
    X, Y = np.meshgrid(xs, ys)
    r_pad = r + 0.5 * res
    mask = (X * X + Y * Y) <= (r_pad * r_pad)
    grid[iy0 : iy1 + 1, ix0 : ix1 + 1] |= mask.astype(np.uint8)


def _inflate_grid(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    H, W = grid.shape
    out = np.zeros_like(grid)
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return out
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                ny = np.clip(ys + dy, 0, H - 1)
                nx = np.clip(xs + dx, 0, W - 1)
                out[ny, nx] = 1
    return out


def _is_connected(inflated: np.ndarray, ox: float, oy: float, res: float, start_xy, goal_xy) -> bool:
    H, W = inflated.shape
    sx = int(np.clip((start_xy[0] - ox) / res, 0, W - 1))
    sy = int(np.clip((start_xy[1] - oy) / res, 0, H - 1))
    gx = int(np.clip((goal_xy[0] - ox) / res, 0, W - 1))
    gy = int(np.clip((goal_xy[1] - oy) / res, 0, H - 1))
    if inflated[sy, sx] or inflated[gy, gx]:
        return False
    vis = np.zeros((H, W), dtype=bool)
    q = deque([(sx, sy)])
    vis[sy, sx] = True
    while q:
        x, y = q.popleft()
        if abs(x - gx) <= 2 and abs(y - gy) <= 2:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not vis[ny, nx] and not inflated[ny, nx]:
                vis[ny, nx] = True
                q.append((nx, ny))
    return False


def _gen_perfect_maze(rng: np.random.Generator, N: int):
    """Recursive backtracker returning v_walls[N, N-1] and h_walls[N-1, N]."""
    v_walls = np.ones((N, N - 1), dtype=bool)
    h_walls = np.ones((N - 1, N), dtype=bool)
    vis = np.zeros((N, N), dtype=bool)
    stack = [(0, 0)]
    vis[0, 0] = True
    while stack:
        x, y = stack[-1]
        nbrs = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not vis[ny, nx]:
                nbrs.append((nx, ny, dx, dy))
        if nbrs:
            nx, ny, dx, dy = nbrs[int(rng.integers(0, len(nbrs)))]
            if dx == 1:
                v_walls[y, x] = False
            elif dx == -1:
                v_walls[y, x - 1] = False
            elif dy == 1:
                h_walls[y, x] = False
            else:
                h_walls[y - 1, x] = False
            vis[ny, nx] = True
            stack.append((nx, ny))
        else:
            stack.pop()
    return v_walls, h_walls


def build_unified_scene(
    buggy_xml_path: str,
    map_type: str,
    terrain_mode: str,
    seed: int,
    grid_res: float = 0.04,
):
    """
    Builds one of 4 procedural multi-terrain scenes and returns:
      (model, data, scene_meta)
    """
    rng = np.random.default_rng(seed)
    wall_h = 0.18
    wall_t = 0.038
    half_t = 0.5 * wall_t

    geoms = []
    boxes_2d = []   # (cx, cy, hx, hy, yaw)
    cyls_2d = []    # (cx, cy, r)

    def add_wall_box(name, cx, cy, hx, hy, yaw=0.0, rgba="0.16 0.18 0.22 1"):
        geoms.append(
            f'<geom name="wall_{name}" type="box" pos="{cx:.4f} {cy:.4f} {0.5*wall_h:.4f}" '
            f'euler="0 0 {math.degrees(yaw):.2f}" size="{hx:.4f} {hy:.4f} {0.5*wall_h:.4f}" rgba="{rgba}"/>'
        )
        boxes_2d.append((cx, cy, hx, hy, yaw))

    def add_pillar(name, cx, cy, r, rgba="0.90 0.45 0.14 1"):
        geoms.append(
            f'<geom name="wall_pil_{name}" type="cylinder" pos="{cx:.4f} {cy:.4f} {0.5*wall_h:.4f}" '
            f'size="{r:.4f} {0.5*wall_h:.4f}" rgba="{rgba}"/>'
        )
        cyls_2d.append((cx, cy, r))

    if map_type in ("ortho_maze", "hybrid_arena"):
        N = int(rng.choice([5, 6, 7])) if map_type == "ortho_maze" else int(rng.choice([6, 7]))
        cs = float(rng.uniform(0.42, 0.54))
        span = N * cs
        xmin, xmax, ymin, ymax = 0.0, span, 0.0, span
        v_walls, h_walls = _gen_perfect_maze(rng, N)

        # For hybrid_arena, carve out a 2x2 or 3x3 open room inside the maze + add 1-2 pillars
        room_cells = set()
        if map_type == "hybrid_arena":
            rw = int(rng.choice([2, 3]))
            rx0 = int(rng.integers(1, max(2, N - rw)))
            ry0 = int(rng.integers(1, max(2, N - rw)))
            for y in range(ry0, ry0 + rw):
                for x in range(rx0, rx0 + rw):
                    room_cells.add((x, y))
                    if x + 1 < rx0 + rw:
                        v_walls[y, x] = False
                    if y + 1 < ry0 + rw:
                        h_walls[y, x] = False
            # Add 1-2 centered pillars inside the carved open room
            n_pil = 1 if rw == 2 else 2
            for pi in range(n_pil):
                px = (rx0 + 0.5 * rw + (0.25 if (n_pil == 2 and pi == 0) else (-0.25 if n_pil == 2 else 0.0))) * cs
                py = (ry0 + 0.5 * rw + rng.uniform(-0.15, 0.15) * cs) * cs
                add_pillar(f"hyb_{pi}", px, py, float(rng.uniform(0.055, 0.080)))

        # Perimeter walls
        mid = 0.5 * span
        add_wall_box("ow", 0.0, mid, half_t, mid + half_t)
        add_wall_box("oe", span, mid, half_t, mid + half_t)
        add_wall_box("os", mid, 0.0, mid + half_t, half_t)
        add_wall_box("on", mid, span, mid + half_t, half_t)

        # Internal walls
        for y in range(N):
            for x in range(N - 1):
                if v_walls[y, x]:
                    add_wall_box(f"v_{x}_{y}", (x + 1) * cs, (y + 0.5) * cs, half_t, 0.5 * cs + half_t)
        for y in range(N - 1):
            for x in range(N):
                if h_walls[y, x]:
                    add_wall_box(f"h_{x}_{y}", (x + 0.5) * cs, (y + 1) * cs, 0.5 * cs + half_t, half_t)

        start_xy = (0.5 * cs, 0.5 * cs)
        goal_xy = ((N - 0.5) * cs, (N - 0.5) * cs)
        init_yaw = 0.0 if not v_walls[0, 0] else 0.5 * math.pi

        zone_centers = _sample_zone_centers(rng, xmin, xmax, ymin, ymax, terrain_mode)
        for y in range(N):
            for x in range(N):
                cx, cy = (x + 0.5) * cs, (y + 0.5) * cs
                tname = _lookup_zone(cx, cy, zone_centers)
                rgba, fric, _ = TERRAIN_SPECS[tname]
                geoms.append(
                    f'<geom name="tile_{x}_{y}" type="box" pos="{cx:.4f} {cy:.4f} -0.025" '
                    f'size="{0.5*cs:.4f} {0.5*cs:.4f} 0.025" rgba="{rgba}" friction="{fric}"/>'
                )

    elif map_type == "curve_track":
        # Continuous curved corridor: random clothoid, 180-deg hairpin, chicane, or dynamic pinch
        subtype = str(rng.choice(["random_curve", "hairpin", "chicane", "pinch"]))
        base_w = float(rng.uniform(0.40, 0.48))
        n_pts = 56
        ds = 0.14
        pts = np.zeros((n_pts, 2), dtype=np.float64)
        headings = np.zeros(n_pts, dtype=np.float64)
        widths = np.full(n_pts, base_w, dtype=np.float64)

        curv = np.zeros(n_pts, dtype=np.float64)
        if subtype == "hairpin":
            sgn = float(rng.choice([-1.0, 1.0]))
            curv[10:26] = sgn * (math.pi / (16 * ds))
            curv[34:50] = -sgn * (math.pi / (16 * ds))
        elif subtype == "chicane":
            amp = float(rng.uniform(1.1, 1.65))
            freq = float(rng.uniform(0.24, 0.34))
            curv = amp * np.sin(np.arange(n_pts) * freq)
        elif subtype == "pinch":
            amp = float(rng.uniform(0.7, 1.2))
            curv = amp * np.sin(np.arange(n_pts) * 0.22 + rng.uniform(0, math.pi))
            widths = base_w - 0.10 * np.exp(-0.5 * ((np.arange(n_pts) - 28) / 6.0) ** 2)
        else:
            # Smooth random spline curvature
            knots = rng.uniform(-1.35, 1.35, size=6)
            curv = np.interp(np.linspace(0, 5, n_pts), np.arange(6), knots)
            curv[:5] = 0.0

        for i in range(1, n_pts):
            headings[i] = headings[i - 1] + curv[i - 1] * ds
            pts[i, 0] = pts[i - 1, 0] + ds * math.cos(headings[i - 1])
            pts[i, 1] = pts[i - 1, 1] + ds * math.sin(headings[i - 1])

        # Build left/right wall segments along centerline
        for i in range(n_pts - 1):
            p0, p1 = pts[i], pts[i + 1]
            seg_vec = p1 - p0
            seg_len = float(np.linalg.norm(seg_vec))
            seg_yaw = math.atan2(seg_vec[1], seg_vec[0])
            nx, ny = -math.sin(seg_yaw), math.cos(seg_yaw)
            half_w = 0.5 * widths[i] + half_t
            mid_p = 0.5 * (p0 + p1)
            add_wall_box(f"cl_{i}", mid_p[0] + nx * half_w, mid_p[1] + ny * half_w, 0.5 * seg_len + 0.025, half_t, seg_yaw)
            add_wall_box(f"cr_{i}", mid_p[0] - nx * half_w, mid_p[1] - ny * half_w, 0.5 * seg_len + 0.025, half_t, seg_yaw)

        # Start and end wall caps
        s_yaw, e_yaw = headings[0], headings[-1]
        add_wall_box("c_start", pts[0, 0] - 0.14 * math.cos(s_yaw), pts[0, 1] - 0.14 * math.sin(s_yaw), half_t, 0.5 * widths[0] + wall_t, s_yaw)
        add_wall_box("c_end", pts[-1, 0] + 0.18 * math.cos(e_yaw), pts[-1, 1] + 0.18 * math.sin(e_yaw), half_t, 0.5 * widths[-1] + wall_t, e_yaw)

        start_xy = (float(pts[1, 0]), float(pts[1, 1]))
        goal_xy = (float(pts[-2, 0]), float(pts[-2, 1]))
        init_yaw = float(headings[1])

        pad = 0.55
        xmin, xmax = float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad)
        ymin, ymax = float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad)
        zone_centers = _sample_zone_centers(rng, xmin, xmax, ymin, ymax, terrain_mode)

        tile_s = 0.55
        nx_t = max(2, int(math.ceil((xmax - xmin) / tile_s)))
        ny_t = max(2, int(math.ceil((ymax - ymin) / tile_s)))
        tw = (xmax - xmin) / nx_t
        th = (ymax - ymin) / ny_t
        for iy in range(ny_t):
            for ix in range(nx_t):
                cx = xmin + (ix + 0.5) * tw
                cy = ymin + (iy + 0.5) * th
                tname = _lookup_zone(cx, cy, zone_centers)
                rgba, fric, _ = TERRAIN_SPECS[tname]
                geoms.append(
                    f'<geom name="tile_{ix}_{iy}" type="box" pos="{cx:.4f} {cy:.4f} -0.025" '
                    f'size="{0.5*tw:.4f} {0.5*th:.4f} 0.025" rgba="{rgba}" friction="{fric}"/>'
                )

    else:
        # open_arena: 3.0m - 4.0m open room with pillars, boxes, and partial baffle walls
        span = float(rng.uniform(3.0, 4.0))
        xmin, xmax, ymin, ymax = 0.0, span, 0.0, span
        mid = 0.5 * span
        add_wall_box("ow", 0.0, mid, half_t, mid + half_t)
        add_wall_box("oe", span, mid, half_t, mid + half_t)
        add_wall_box("os", mid, 0.0, mid + half_t, half_t)
        add_wall_box("on", mid, span, mid + half_t, half_t)

        start_xy = (0.38, 0.38)
        goal_xy = (span - 0.38, span - 0.38)
        init_yaw = float(rng.uniform(0.0, 0.5 * math.pi))

        # Place 5-9 random obstacles (cylinders, boxes, short baffles) clear of start & goal
        n_obs = int(rng.integers(5, 10))
        placed = []
        for oi in range(n_obs):
            for _ in range(25):
                ox_c = float(rng.uniform(0.55, span - 0.55))
                oy_c = float(rng.uniform(0.55, span - 0.55))
                if math.hypot(ox_c - start_xy[0], oy_c - start_xy[1]) < 0.62:
                    continue
                if math.hypot(ox_c - goal_xy[0], oy_c - goal_xy[1]) < 0.62:
                    continue
                if any(math.hypot(ox_c - px, oy_c - py) < 0.58 for px, py in placed):
                    continue
                placed.append((ox_c, oy_c))
                kind = int(rng.integers(0, 3))
                if kind == 0:
                    add_pillar(f"op_{oi}", ox_c, oy_c, float(rng.uniform(0.08, 0.14)))
                elif kind == 1:
                    s_box = float(rng.uniform(0.09, 0.15))
                    add_wall_box(f"ob_{oi}", ox_c, oy_c, s_box, s_box, float(rng.uniform(0, math.pi)), rgba="0.88 0.42 0.16 1")
                else:
                    blen = float(rng.uniform(0.28, 0.48))
                    add_wall_box(f"baf_{oi}", ox_c, oy_c, blen, half_t, float(rng.choice([0.0, 0.5 * math.pi, 0.25 * math.pi])))
                break

        zone_centers = _sample_zone_centers(rng, xmin, xmax, ymin, ymax, terrain_mode)
        n_tiles = 6
        ts = span / n_tiles
        for iy in range(n_tiles):
            for ix in range(n_tiles):
                cx, cy = (ix + 0.5) * ts, (iy + 0.5) * ts
                tname = _lookup_zone(cx, cy, zone_centers)
                rgba, fric, _ = TERRAIN_SPECS[tname]
                geoms.append(
                    f'<geom name="tile_{ix}_{iy}" type="box" pos="{cx:.4f} {cy:.4f} -0.025" '
                    f'size="{0.5*ts:.4f} {0.5*ts:.4f} 0.025" rgba="{rgba}" friction="{fric}"/>'
                )

    # Start & Goal visual markers (non-colliding)
    geoms.append(f'<geom name="start_pad" type="cylinder" pos="{start_xy[0]:.4f} {start_xy[1]:.4f} 0.001" size="0.16 0.001" contype="0" conaffinity="0" rgba="0.12 0.78 0.36 0.85"/>')
    geoms.append(f'<geom name="goal_pad" type="cylinder" pos="{goal_xy[0]:.4f} {goal_xy[1]:.4f} 0.001" size="0.18 0.001" contype="0" conaffinity="0" rgba="0.92 0.20 0.22 0.90"/>')

    # Rasterize 2D occupancy grid for Phase 3 universal A* + Pure Pursuit teacher
    margin = 0.20
    ox, oy = xmin - margin, ymin - margin
    W = int(math.ceil((xmax - xmin + 2 * margin) / grid_res))
    H = int(math.ceil((ymax - ymin + 2 * margin) / grid_res))
    occ_grid = np.zeros((H, W), dtype=np.uint8)
    for cx, cy, hx, hy, yaw in boxes_2d:
        _rasterize_box(occ_grid, ox, oy, grid_res, cx, cy, hx, hy, yaw)
    for cx, cy, r in cyls_2d:
        _rasterize_cyl(occ_grid, ox, oy, grid_res, cx, cy, r)

    scene_xml = f"""
    <mujoco model="unified_buggy_2_{map_type}">
      <option timestep="0.005" integrator="RK4"/>
      <visual><headlight ambient="0.52 0.52 0.52" diffuse="0.65 0.65 0.65"/></visual>
      <worldbody>
        <light pos="{0.5*(xmin+xmax):.2f} {0.5*(ymin+ymax):.2f} 6.5" dir="0 0 -1"/>
        {''.join(geoms)}
      </worldbody>
    </mujoco>
    """

    spec = mujoco.MjSpec.from_string(scene_xml)
    bot_spec = mujoco.MjSpec.from_file(buggy_xml_path)
    q_init = [math.cos(0.5 * init_yaw), 0.0, 0.0, math.sin(0.5 * init_yaw)]
    frame = spec.worldbody.add_frame(pos=[start_xy[0], start_xy[1], 0.035], quat=q_init)
    if hasattr(spec, "attach"):
        spec.attach(bot_spec, prefix="bot/", frame=frame)
    else:
        frame.attach_body(bot_spec.worldbody.first_body(), "bot/", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    meta = {
        "map_type": map_type,
        "seed": seed,
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "init_yaw": init_yaw,
        "bounds": (xmin, xmax, ymin, ymax),
        "max_diag": math.hypot(xmax - xmin, ymax - ymin),
        "zone_centers": zone_centers,
        "occ_grid": occ_grid,
        "occ_origin": (ox, oy),
        "occ_res": grid_res,
    }
    return model, data, meta


class UnifiedBuggyEnv(gym.Env):
    """
    Phase 2 Unified Multi-Geometry, Multi-Terrain Continuous Environment for RL_buggy_2.0.
      - Obs: 28-D (16-ray LIDAR + kinematics + Euclidean goal + 4-sector odometry memory)
      - Act: 2-D continuous Box([-1, 1]) -> v in [0.0, 0.48] m/s, w in [-2.6, 2.6] rad/s
      - Inner loop: Frozen 200 Hz leaky Z-gyro PI (identical across train & eval)
    """

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy_v2_lidar.xml",
        map_mode: str = "mixed",
        terrain_mode: str = "multi_zone",
        max_steps: int = 900,
        noise_enabled: bool = True,
        render_realtime: bool = False,
        base_seed: int = 0,
        map_probs=None,
    ):
        super().__init__()
        self.buggy_xml_path = buggy_xml_path
        self.map_mode = map_mode
        self.map_probs = tuple(map_probs) if map_probs is not None else (0.30, 0.25, 0.25, 0.20)
        self.terrain_mode = terrain_mode
        self.max_steps = max_steps
        self.noise_enabled = noise_enabled
        self.render_realtime = render_realtime
        self.rng = np.random.default_rng(base_seed)

        self.wheel_radius = 0.032
        self.track_width = 0.135
        self.max_wheel_vel = 22.0
        self.ticks_per_step = 20  # 10 Hz policy over 200 Hz physics (dt=0.005)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)

        self.sensors = PlanarLidarSensorSuite(
            num_rays=16,
            max_range=2.0,
            wheel_radius=self.wheel_radius,
            track_width=self.track_width,
            noise_enabled=noise_enabled,
        )

        self.model = None
        self.data = None
        self.meta = None
        self.viewer = None
        self._act_l = 0
        self._act_r = 1
        self._wall_geom_ids = set()
        self._bot_geom_ids = set()

        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_v_norm = -1.0
        self._prev_w_norm = 0.0
        self._prev_euc = 0.0
        self._visited_cells = set()

    def _bind_ids(self):
        self.sensors.bind(self.model, prefix="bot/")

        # Bind wheel actuators
        self._act_l, self._act_r = 0, 1
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            low = name.lower()
            if "left" in low or low.endswith("_l"):
                self._act_l = i
            elif "right" in low or low.endswith("_r"):
                self._act_r = i

        # Cache wall vs bot geom IDs for exact collision counting
        self._wall_geom_ids = set()
        self._bot_geom_ids = set()
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("wall_"):
                self._wall_geom_ids.add(g)
            elif name.startswith("bot/"):
                self._bot_geom_ids.add(g)

    def _count_wall_contacts(self) -> int:
        hits = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._bot_geom_ids and g2 in self._wall_geom_ids) or (
                g2 in self._bot_geom_ids and g1 in self._wall_geom_ids
            ):
                hits += 1
        return hits

    def body_pose(self):
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return bx, by, yaw

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = int(seed) if seed is not None else int(self.rng.integers(0, 1000000))
        local_rng = np.random.default_rng(ep_seed)

        if self.map_mode == "mixed":
            map_type = str(
                local_rng.choice(
                    ["ortho_maze", "curve_track", "open_arena", "hybrid_arena"],
                    p=self.map_probs,
                )
            )
        else:
            map_type = self.map_mode

        # Retry open_arena if random obstacles ever seal off start->goal
        for attempt in range(8):
            self.model, self.data, self.meta = build_unified_scene(
                self.buggy_xml_path,
                map_type=map_type,
                terrain_mode=self.terrain_mode,
                seed=ep_seed + attempt * 997,
            )
            infl = _inflate_grid(self.meta["occ_grid"], radius_cells=3)
            if _is_connected(infl, self.meta["occ_origin"][0], self.meta["occ_origin"][1], self.meta["occ_res"], self.meta["start_xy"], self.meta["goal_xy"]):
                break

        self._bind_ids()
        self.steps = 0
        self.wall_hits = 0
        self._consec_hits = 0
        self._cur_v = 0.0
        self._cur_w = 0.0
        self._w_err_int = 0.0
        self._prev_v_norm = -1.0
        self._prev_w_norm = 0.0

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        bx, by, yaw = self.body_pose()
        self.sensors.reset_odom(bx, by, yaw)
        obs, _, euc_dist = self.sensors.build_obs(
            self.data, self.rng, self.meta["goal_xy"], self.meta["max_diag"], self._prev_v_norm, self._prev_w_norm
        )
        self._prev_euc = euc_dist
        self._visited_cells = {(int(round(bx / 0.35)), int(round(by / 0.35)))}

        return obs, {
            "map_type": map_type,
            "seed": ep_seed,
            "euc_dist": euc_dist,
        }

    def step(self, action):
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        v_norm = float(np.clip(act[0], -1.0, 1.0))
        w_norm = float(np.clip(act[1], -1.0, 1.0))

        v_cmd = 0.5 * (v_norm + 1.0) * 0.48
        w_cmd = w_norm * 2.60

        self.steps += 1
        dt = self.model.opt.timestep
        max_dv = 3.0 * dt
        max_dw = 18.0 * dt
        step_hits = 0
        cur_terrain = "pavement"

        for _ in range(self.ticks_per_step):
            bx, by, _ = self.body_pose()
            cur_terrain = _lookup_zone(bx, by, self.meta["zone_centers"])
            drag = TERRAIN_SPECS[cur_terrain][2]

            # Clean, frozen 200 Hz leaky Z-gyro PI (no eval-time hacks)
            gz = self.sensors.read_gyro_z(self.data, self.rng)
            enc_v, _ = self.sensors.read_wheel_kinematics(self.data, self.rng)
            self.sensors.update_odom(enc_v, gz, dt)

            w_err = w_cmd - gz
            self._w_err_int = float(np.clip(0.95 * self._w_err_int + w_err * dt, -0.35, 0.35))
            w_target = float(np.clip(w_cmd + 0.75 * w_err + 1.40 * self._w_err_int, -4.0, 4.0))

            self._cur_v += float(np.clip(v_cmd - self._cur_v, -max_dv, max_dv))
            self._cur_w += float(np.clip(w_target - self._cur_w, -max_dw, max_dw))

            w_l = (self._cur_v - 0.5 * self._cur_w * self.track_width) / self.wheel_radius
            w_r = (self._cur_v + 0.5 * self._cur_w * self.track_width) / self.wheel_radius
            self.data.ctrl[self._act_l] = float(np.clip(w_l, -self.max_wheel_vel, self.max_wheel_vel))
            self.data.ctrl[self._act_r] = float(np.clip(w_r, -self.max_wheel_vel, self.max_wheel_vel))

            if drag > 0.0:
                vx, vy, wz = float(self.data.qvel[0]), float(self.data.qvel[1]), float(self.data.qvel[5])
                self.data.qfrc_applied[0] = -drag * vx * (1.0 + 2.0 * abs(vx))
                self.data.qfrc_applied[1] = -drag * vy * (1.0 + 2.0 * abs(vy))
                self.data.qfrc_applied[5] = -0.08 * drag * wz
            else:
                self.data.qfrc_applied[:6] = 0.0

            if self.viewer is not None and hasattr(self.viewer, "lock"):
                with self.viewer.lock():
                    mujoco.mj_step(self.model, self.data)
            else:
                mujoco.mj_step(self.model, self.data)

            if self._count_wall_contacts() > 0:
                step_hits += 1
                self.wall_hits += 1

            if self.viewer is not None:
                self.viewer.sync()
                if self.render_realtime:
                    time.sleep(dt)

        obs, lidar_m, euc_dist = self.sensors.build_obs(
            self.data, self.rng, self.meta["goal_xy"], self.meta["max_diag"], v_norm, w_norm
        )
        at_goal = euc_dist < 0.24

        # Dense reward for Stage B PPO: exploration + speed-scaled forward pillar/doorway margin
        bx, by, _ = self.body_pose()
        vcell = (int(round(bx / 0.35)), int(round(by / 0.35)))
        novel_bonus = 0.30 if vcell not in self._visited_cells else 0.0
        self._visited_cells.add(vcell)

        euc_delta = self._prev_euc - euc_dist
        self._prev_euc = euc_dist

        min_clear = float(lidar_m.min())
        front_cone = float(min(lidar_m[0], lidar_m[1], lidar_m[2], lidar_m[14], lidar_m[15]))
        side_pen = -0.30 * max(0.0, 0.105 - min_clear) / 0.105
        front_pen = -0.35 * (v_cmd / 0.48) * max(0.0, 0.18 - front_cone) / 0.18
        smooth_pen = -0.01 * abs(w_norm - self._prev_w_norm)

        reward = 2.0 * euc_delta + novel_bonus + side_pen + front_pen + smooth_pen - 0.012

        if step_hits > 0:
            self._consec_hits += 1
            reward -= 2.0
        else:
            self._consec_hits = 0

        crashed = (self._consec_hits >= 5) or (self.wall_hits >= 25)
        if crashed:
            reward -= 18.0
        if at_goal:
            reward += 35.0 + (10.0 if self.wall_hits == 0 else 0.0)

        self._prev_v_norm = v_norm
        self._prev_w_norm = w_norm
        terminated = at_goal or crashed
        truncated = (not terminated) and (self.steps >= self.max_steps)

        info = {
            "is_success": at_goal,
            "crashed": crashed,
            "map_type": self.meta["map_type"],
            "terrain": cur_terrain,
            "wall_hits": self.wall_hits,
            "euc_dist": euc_dist,
            "min_lidar": min_clear,
        }
        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    if "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        mtype = args[0] if args else "hybrid_arena"
        seed = int(args[1]) if len(args) > 1 else 42
        env = UnifiedBuggyEnv(map_mode=mtype, terrain_mode="multi_zone", render_realtime=True)
        obs, info = env.reset(seed=seed)
        xmin, xmax, ymin, ymax = env.meta["bounds"]
        print(f"PREVIEW | map={info['map_type']} | seed={seed} | obs_shape={obs.shape} | euc={info['euc_dist']:.2f}m")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            env.viewer = viewer
            viewer.cam.lookat[:] = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0]
            viewer.cam.distance = max(xmax - xmin, ymax - ymin) * 1.35
            viewer.cam.elevation = -74.0
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    else:
        print("=== Phase 2 UnifiedBuggyEnv Verification (All 4 Map Families) ===")
        for mtype in ("ortho_maze", "curve_track", "open_arena", "hybrid_arena"):
            env = UnifiedBuggyEnv(map_mode=mtype, terrain_mode="multi_zone", noise_enabled=True)
            obs, info = env.reset(seed=100)
            assert obs.shape == (28,) and np.all(np.isfinite(obs)), f"Bad obs in {mtype}"
            infl = _inflate_grid(env.meta["occ_grid"], radius_cells=3)
            conn = _is_connected(infl, env.meta["occ_origin"][0], env.meta["occ_origin"][1], env.meta["occ_res"], env.meta["start_xy"], env.meta["goal_xy"])
            for _ in range(15):
                obs, rew, tm, tc, step_info = env.step(np.array([0.4, 0.0], dtype=np.float32))
            zones = sorted({z[2] for z in env.meta["zone_centers"]})
            print(
                f"  {mtype:13s} | obs={obs.shape} | connected={conn} | "
                f"occ={env.meta['occ_grid'].shape} | min_lidar={step_info['min_lidar']:.3f}m | zones={zones}"
            )
        print("\nPASS: Phase 2 UnifiedBuggyEnv verified across all 4 procedural map families.")
