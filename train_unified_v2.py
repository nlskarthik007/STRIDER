import os
import sys
import math
import heapq
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.unified_buggy_env import UnifiedBuggyEnv


MAP_FAMILIES = ("ortho_maze", "curve_track", "open_arena", "hybrid_arena")
DAGGER_CKPT = "checkpoints/unified_lidar_v2_dagger.zip"
PPO_CKPT = "checkpoints/unified_lidar_v2_ppo.zip"


class UniversalCenterlineTeacher:
    """
    Universal continuous (v, w) teacher across ortho mazes, curves/hairpins,
    open obstacle arenas, and hybrid maps via clearance-weighted Dijkstra + Pure Pursuit.
    """

    def __init__(self):
        self.cost_field = None
        self.dist_wall = None
        self.ox = 0.0
        self.oy = 0.0
        self.res = 0.04
        self.H = 0
        self.W = 0

    def reset(self, env: UnifiedBuggyEnv):
        occ = env.meta["occ_grid"]
        self.ox, self.oy = env.meta["occ_origin"]
        self.res = env.meta["occ_res"]
        self.H, self.W = occ.shape

        # 1. Multi-source BFS distance-to-wall field (in grid cells)
        dist = np.full((self.H, self.W), 999, dtype=np.int16)
        q = deque()
        ys, xs = np.nonzero(occ)
        for y, x in zip(ys, xs):
            dist[y, x] = 0
            q.append((int(x), int(y)))

        while q:
            x, y = q.popleft()
            nd = dist[y, x] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and dist[ny, nx] > nd:
                    dist[ny, nx] = nd
                    q.append((nx, ny))
        self.dist_wall = dist

        # 2. Clearance-weighted 8-connected Dijkstra from goal_xy
        gx, gy = env.meta["goal_xy"]
        gix = int(np.clip((gx - self.ox) / self.res, 0, self.W - 1))
        giy = int(np.clip((gy - self.oy) / self.res, 0, self.H - 1))

        # Penalize near-wall cells so steepest descent rides corridor/curve centerlines
        cell_pen = np.zeros((self.H, self.W), dtype=np.float32)
        cell_pen[dist <= 1] = 1e6
        cell_pen[dist == 2] = 250.0
        cell_pen[dist == 3] = 60.0
        cell_pen[dist == 4] = 20.0
        cell_pen[dist == 5] = 7.0
        cell_pen[dist == 6] = 2.5
        cell_pen[dist == 7] = 0.8
        cell_pen[dist == 8] = 0.25

        cost = np.full((self.H, self.W), np.inf, dtype=np.float32)
        cost[giy, gix] = 0.0
        pq = [(0.0, gix, giy)]

        nbrs = (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (-1, -1, 1.4142),
        )
        while pq:
            c, x, y = heapq.heappop(pq)
            if c > cost[y, x]:
                continue
            for dx, dy, step_len in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and dist[ny, nx] > 1:
                    nc = c + step_len + float(cell_pen[ny, nx])
                    if nc < cost[ny, nx]:
                        cost[ny, nx] = nc
                        heapq.heappush(pq, (nc, nx, ny))

        self.cost_field = cost

    def _has_clear_los(self, x0: float, y0: float, x1: float, y1: float, min_clear_cells: int = 2) -> bool:
        """Check that the straight chord from (x0, y0) to (x1, y1) does not clip an inner corner."""
        dist_m = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(math.ceil(dist_m / (0.5 * self.res))))
        for k in range(1, steps + 1):
            t = k / steps
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            ix = int(np.clip((px - self.ox) / self.res, 0, self.W - 1))
            iy = int(np.clip((py - self.oy) / self.res, 0, self.H - 1))
            if self.dist_wall[iy, ix] <= min_clear_cells:
                return False
        return True

    def _trace_lookahead(self, bx: float, by: float, lookahead_m: float = 0.26):
        ix = int(np.clip((bx - self.ox) / self.res, 0, self.W - 1))
        iy = int(np.clip((by - self.oy) / self.res, 0, self.H - 1))

        if not np.isfinite(self.cost_field[iy, ix]):
            best_c = np.inf
            best_xy = (ix, iy)
            for rad in range(1, 7):
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            val = self.cost_field[ny, nx]
                            if np.isfinite(val) and val < best_c:
                                best_c = val
                                best_xy = (nx, ny)
                if np.isfinite(best_c):
                    break
            ix, iy = best_xy

        cx, cy = ix, iy
        path_pts = [(self.ox + (cx + 0.5) * self.res, self.oy + (cy + 0.5) * self.res)]
        travelled = 0.0

        # 1. Trace full centerline segment along steepest descent
        for _ in range(14):
            best_c = self.cost_field[cy, cx]
            nxt = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        val = self.cost_field[ny, nx]
                        if val < best_c:
                            best_c = val
                            nxt = (nx, ny)
            if nxt is None:
                break
            travelled += math.hypot(nxt[0] - cx, nxt[1] - cy) * self.res
            cx, cy = nxt
            path_pts.append((self.ox + (cx + 0.5) * self.res, self.oy + (cy + 0.5) * self.res))
            if travelled >= lookahead_m:
                break

        # 2. Pick furthest point on the traced path with STRICT wall-clear line of sight
        cur_clear = int(self.dist_wall[iy, ix])
        min_clear = 3 if cur_clear >= 4 else (2 if cur_clear >= 3 else 1)
        los_idx = 0
        for k in range(1, len(path_pts)):
            px, py = path_pts[k]
            if k >= 2 and not self._has_clear_los(bx, by, px, py, min_clear_cells=min_clear):
                break
            los_idx = k

        tx, ty = path_pts[los_idx]
        dist_to_los = math.hypot(tx - bx, ty - by)

        # 3. If we are already sitting at the corner center (LOS blocked immediately ahead),
        #    return the post-corner tangent direction so the buggy pivots cleanly into the new corridor
        if los_idx < len(path_pts) - 1 and dist_to_los < 0.065:
            far_x, far_y = path_pts[min(len(path_pts) - 1, los_idx + 4)]
            return far_x, far_y, True

        return tx, ty, False

    def act(self, env: UnifiedBuggyEnv) -> np.ndarray:
        bx, by, yaw = env.body_pose()
        tx, ty, at_corner_pivot = self._trace_lookahead(bx, by, lookahead_m=0.26)

        dx, dy = tx - bx, ty - by
        desired_yaw = math.atan2(dy, dx)
        alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi

        lidar = env.sensors.read_lidar(env.data, env.rng)
        front_c = float(min(lidar[0], lidar[1] + 0.02, lidar[15] + 0.02))
        fl_c = float(min(lidar[1], lidar[2]))
        fr_c = float(min(lidar[15], lidar[14]))
        side_min = float(min(lidar[2], lidar[3], lidar[4], lidar[12], lidar[13], lidar[14]))

        # Crisp zero-radius pivot at 90-deg corner center, dead-end, or wall face
        if abs(alpha) > 0.90 or (at_corner_pivot and abs(alpha) > 0.30) or (front_c < 0.105 and abs(alpha) > 0.22):
            v_des = 0.0
            w_des = float(np.clip(3.2 * alpha, -2.40, 2.40))
            if abs(w_des) < 0.85:
                w_des = math.copysign(0.85, alpha if abs(alpha) <= 2.45 else 1.0)
        else:
            v_base = 0.44 * (max(0.0, math.cos(alpha)) ** 1.5)
            front_scale = float(np.clip((front_c - 0.085) / 0.22, 0.0, 1.0))
            side_scale = float(np.clip((side_min - 0.060) / 0.08, 0.40, 1.0))
            v_des = v_base * min(front_scale, side_scale)
            if front_c > 0.115 and abs(alpha) < 0.65:
                v_des = max(v_des, 0.09)

            sh_l = float(min(lidar[1], lidar[2], lidar[3]))
            sh_r = float(min(lidar[15], lidar[14], lidar[13]))
            rep = 0.0
            if sh_l < 0.145:
                rep -= 5.5 * (0.145 - sh_l)
            if sh_r < 0.145:
                rep += 5.5 * (0.145 - sh_r)
            if min(sh_l, sh_r) < 0.115:
                v_des = min(v_des, 0.20)
            w_des = float(np.clip(2.85 * alpha + rep, -2.60, 2.60))

        v_norm = float(np.clip(2.0 * (v_des / 0.48) - 1.0, -1.0, 1.0))
        w_norm = float(np.clip(w_des / 2.60, -1.0, 1.0))
        return np.array([v_norm, w_norm], dtype=np.float32)


def eval_unified(model: RecurrentPPO, seeds=range(9800, 9810), verbose: bool = True):
    model.policy.eval()
    tot_w, tot_zh, tot_n = 0, 0, 0
    for mtype in MAP_FAMILIES:
        env = UnifiedBuggyEnv(map_mode=mtype, terrain_mode="multi_zone", max_steps=900, noise_enabled=True)
        w, zh, t_list, h_list = 0, 0, [], []
        for s in seeds:
            obs, _ = env.reset(seed=s)
            st, ep, done = None, np.ones((1,), dtype=bool), False
            while not done:
                a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
                ep = np.zeros((1,), dtype=bool)
                obs, _, tm, tc, info = env.step(a)
                done = tm or tc
            w += int(info["is_success"])
            zh += int(info["is_success"] and info["wall_hits"] == 0)
            t_list.append(env.steps * 0.1)
            h_list.append(info["wall_hits"])
        n = len(seeds)
        tot_w += w
        tot_zh += zh
        tot_n += n
        if verbose:
            print(
                f"  {mtype:13s} (multi-zone) | solve: {w:2d}/{n:2d} (0-hit: {zh:2d}) "
                f"| avg time: {np.mean(t_list):5.1f}s | avg hits: {np.mean(h_list):4.1f}"
            )
    if verbose:
        print(f"\nOVERALL UNIFIED V2 SCORE: {tot_w}/{tot_n} ({100.0*tot_w/tot_n:.1f}%) | 0-hit: {tot_zh}/{tot_n}")
    return tot_w, tot_zh, tot_n


def check_teacher(seeds=range(9800, 9805)):
    teacher = UniversalCenterlineTeacher()
    print("=== Universal Centerline Teacher Check (Multi-Zone Terrains, Noise ON) ===")
    tot_w, tot_zh, tot_n = 0, 0, 0
    for mtype in MAP_FAMILIES:
        env = UnifiedBuggyEnv(map_mode=mtype, terrain_mode="multi_zone", max_steps=900, noise_enabled=True)
        w, zh, t_list, h_list = 0, 0, [], []
        for s in seeds:
            env.reset(seed=s)
            teacher.reset(env)
            done = False
            while not done:
                _, _, tm, tc, info = env.step(teacher.act(env))
                done = tm or tc
            w += int(info["is_success"])
            zh += int(info["is_success"] and info["wall_hits"] == 0)
            t_list.append(env.steps * 0.1)
            h_list.append(info["wall_hits"])
        n = len(seeds)
        tot_w += w
        tot_zh += zh
        tot_n += n
        print(
            f"  Teacher {mtype:13s}: {w}/{n} (0-hit: {zh}/{n}) "
            f"| avg time: {np.mean(t_list):4.1f}s | avg hits: {np.mean(h_list):4.1f}"
        )
    print(f"OVERALL TEACHER: {tot_w}/{tot_n} ({100.0*tot_w/tot_n:.1f}%) | 0-hit: {tot_zh}/{tot_n}")


def train_stage_a_dagger(save_path: str = DAGGER_CKPT):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = UnifiedBuggyEnv(map_mode="mixed", terrain_mode="multi_zone", max_steps=900, noise_enabled=True, base_seed=7000, map_probs=(0.24, 0.22, 0.27, 0.27))

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        policy_kwargs=dict(lstm_hidden_size=192, n_lstm_layers=1, net_arch=dict(pi=[128, 64], vf=[128, 64])),
        verbose=0,
    )

    teacher = UniversalCenterlineTeacher()
    rng = np.random.default_rng(2026)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers

    params = [
        p
        for m in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net)
        if m is not None
        for p in m.parameters()
    ]
    opt = torch.optim.Adam(params, lr=1.2e-3)

    betas = (1.00, 0.75, 0.50, 0.30, 0.16, 0.08, 0.03, 0.00, 0.00, 0.00)
    eps_per_round = 48
    epochs_per_round = 14
    lr_start, lr_end = 1.2e-3, 1.5e-4

    episodes = []
    best_score = (-1, -1)

    print(f"=== STAGE A: UNIFIED DAgger TRAINING (8 Rounds x {eps_per_round} eps) ===")
    for r, beta in enumerate(betas):
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * r / (len(betas) - 1)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=7000 + r * 400 + ep)
            teacher.reset(env)
            ep_o, ep_a = [], []
            st, ep_s, done = None, np.ones((1,), dtype=bool), False
            while not done:
                ea = teacher.act(env)
                sa, st = model.predict(obs, state=st, episode_start=ep_s, deterministic=True)
                ep_s = np.zeros((1,), dtype=bool)
                ep_o.append(obs)
                ep_a.append(ea)
                step_act = ea if (rng.random() < beta) else sa
                obs, _, tm, tc, info = env.step(step_act)
                done = tm or tc
            wins += int(info["is_success"])
            if len(ep_o) > 10:
                episodes.append((np.asarray(ep_o, np.float32), np.asarray(ep_a, np.float32)))

        # Keep dataset bounded to most recent 240 episodes to favor on-policy student distribution
        if len(episodes) > 320:
            episodes = episodes[-320:]

        model.policy.train()
        order = np.arange(len(episodes))
        for _ in range(epochs_per_round):
            rng.shuffle(order)
            ls = 0.0
            for idx in order:
                o_np, a_np = episodes[idx]
                starts = torch.zeros(len(o_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)
                dist, _ = model.policy.get_distribution(torch.as_tensor(o_np, device=device), (h0, c0), starts)
                loss = F.smooth_l1_loss(dist.distribution.mean, torch.as_tensor(a_np, device=device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ls += float(loss.item())

        vw, vz, vn = eval_unified(model, seeds=range(9700, 9707), verbose=False)
        tag = ""
        if (vw, vz) >= best_score:
            best_score = (vw, vz)
            model.save(save_path)
            tag = " <-- NEW BEST"

        print(
            f"[Unified DAgger R{r} beta={beta:.2f} lr={lr:.1e}] rollout: {wins:2d}/{eps_per_round} "
            f"| L1: {ls/len(episodes):.4f} | unseen val (20 maps): {vw:2d}/{vn} (0-hit: {vz:2d}){tag}"
        )

    best_model = RecurrentPPO.load(save_path)
    print(f"\n=== STAGE A HELD-OUT BENCHMARK (40 Unseen Multi-Zone Maps) | {save_path} ===")
    eval_unified(best_model, seeds=range(9800, 9810), verbose=True)


def train_stage_b_ppo(
    init_ckpt: str = DAGGER_CKPT,
    save_path: str = PPO_CKPT,
    warmup_timesteps: int = 20_480,
    total_timesteps: int = 120_000,
    eval_every: int = 20_000,
):
    assert os.path.exists(init_ckpt), f"Run Stage A DAgger first — missing {init_ckpt}"

    # Focus 60% of rollouts on open_arena & hybrid_arena while replaying ortho/curve
    env = UnifiedBuggyEnv(
        map_mode="mixed",
        terrain_mode="multi_zone",
        max_steps=900,
        noise_enabled=True,
        base_seed=12000,
        map_probs=(0.20, 0.20, 0.25, 0.35),
    )

    model = RecurrentPPO.load(
        init_ckpt,
        env=env,
        custom_objects={
            "learning_rate": 3.0e-4,
            "clip_range": 0.12,
            "ent_coef": 0.0,
            "n_steps": 512,
            "batch_size": 256,
        },
    )
    model.target_kl = 0.015

    with torch.no_grad():
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data.fill_(-2.1)

    # Snapshot exact Stage A actor weights before critic warmup
    actor_prefixes = ("lstm_actor", "mlp_extractor.policy_net", "action_net", "log_std")
    actor_snapshot = {
        k: v.detach().clone()
        for k, v in model.policy.state_dict().items()
        if k.startswith(actor_prefixes)
    }

    val_seeds = range(9700, 9707)  # 28 unseen multi-zone maps (7 per family)
    best_w, best_z, vn = eval_unified(model, seeds=val_seeds, verbose=False)
    model.save(save_path)
    print(f"=== STAGE B1: CRITIC-ONLY WARMUP ({warmup_timesteps} steps) ===")
    print(f"[PPO Init] unseen val: {best_w}/{vn} (0-hit: {best_z}/{vn})")

    # Freeze actor so random critic gradients cannot corrupt the 10/10 ortho/curve policy
    for name, p in model.policy.named_parameters():
        if name.startswith(actor_prefixes):
            p.requires_grad = False

    model.learn(total_timesteps=warmup_timesteps, reset_num_timesteps=False, progress_bar=False)

    # Restore exact actor weights, unfreeze, reset Adam moments, and lower LR for joint PPO
    model.policy.load_state_dict(actor_snapshot, strict=False)
    for p in model.policy.parameters():
        p.requires_grad = True
    model.policy.optimizer.state.clear()
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = 6.0e-5
    model.lr_schedule = lambda _: 6.0e-5
    model.clip_range = lambda _: 0.12
    model.ent_coef = 0.0015

    print(f"=== STAGE B2: JOINT RECURRENT-PPO FINE-TUNING ({total_timesteps} steps) ===")
    done_steps = 0
    while done_steps < total_timesteps:
        chunk = min(eval_every, total_timesteps - done_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done_steps += chunk

        vw, vz, vn = eval_unified(model, seeds=val_seeds, verbose=False)
        tag = ""
        if (vw, vz) >= (best_w, best_z):
            best_w, best_z = vw, vz
            model.save(save_path)
            tag = " <-- NEW BEST"
        print(f"[PPO {done_steps:6d}/{total_timesteps}] unseen val: {vw:2d}/{vn} (0-hit: {vz:2d}){tag}")

    best_model = RecurrentPPO.load(save_path)
    print(f"\n=== STAGE B HELD-OUT BENCHMARK (40 Unseen Multi-Zone Maps) | {save_path} ===")
    eval_unified(best_model, seeds=range(9800, 9810), verbose=True)


def run_viewer(ckpt: str, map_type: str = "hybrid_arena", seed: int = 9800):
    model = RecurrentPPO.load(ckpt)
    env = UnifiedBuggyEnv(map_mode=map_type, terrain_mode="multi_zone", max_steps=900, noise_enabled=True, render_realtime=True)
    obs, info = env.reset(seed=seed)
    xmin, xmax, ymin, ymax = env.meta["bounds"]
    print(f"LIVE UNIFIED V2 | ckpt={ckpt} | map={info['map_type']} | seed={seed}")
    st, ep = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.0]
        viewer.cam.distance = max(xmax - xmin, ymax - ymin) * 1.35
        viewer.cam.elevation = -74.0
        while viewer.is_running():
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, info = env.step(a)
            if tm or tc:
                status = "GOAL REACHED" if info["is_success"] else "ENDED"
                print(f"{status} | time={env.steps*0.1:.1f}s | terrain={info['terrain']} | wall_hits={info['wall_hits']}")
                time.sleep(1.5)
                break



POLISHED_CKPT = "checkpoints/unified_lidar_v2_polished.zip"


def train_focused_polish(
    init_ckpt: str = "checkpoints/unified_lidar_v2_95pct_0hit.zip",
    save_path: str = POLISHED_CKPT,
):
    assert os.path.exists(init_ckpt), f"Missing base checkpoint: {init_ckpt}"

    env = UnifiedBuggyEnv(
        map_mode="mixed",
        terrain_mode="multi_zone",
        max_steps=900,
        noise_enabled=True,
        base_seed=30000,
        map_probs=(0.20, 0.20, 0.25, 0.35),
    )

    model = RecurrentPPO.load(init_ckpt, env=env)
    teacher = UniversalCenterlineTeacher()
    rng = np.random.default_rng(2028)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers

    params = [
        p
        for m in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net)
        if m is not None
        for p in m.parameters()
    ]
    opt = torch.optim.Adam(params, lr=6.0e-5)

    val_seeds = range(9700, 9707)
    best_w, best_z, vn = eval_unified(model, seeds=val_seeds, verbose=False)
    model.save(save_path)
    print(f"=== GENTLE HYBRID/OPEN DAGGER POLISH (Pre-fill 160 eps + 4 Rounds x 48 eps) ===")
    print(f"[Polish Init] unseen val ({vn} maps): {best_w:2d}/{vn} (0-hit: {best_z:2d})")

    episodes = []

    def collect_rollouts(n_eps: int, seed_base: int):
        wins = 0
        model.policy.eval()
        for ep in range(n_eps):
            obs, _ = env.reset(seed=seed_base + ep)
            teacher.reset(env)
            ep_o, ep_a = [], []
            st, ep_s, done = None, np.ones((1,), dtype=bool), False
            while not done:
                ea = teacher.act(env)
                sa, st = model.predict(obs, state=st, episode_start=ep_s, deterministic=True)
                ep_s = np.zeros((1,), dtype=bool)
                ep_o.append(obs)
                ep_a.append(ea)
                obs, _, tm, tc, info = env.step(sa)
                done = tm or tc
            wins += int(info["is_success"])
            if len(ep_o) > 10:
                episodes.append((np.asarray(ep_o, np.float32), np.asarray(ep_a, np.float32)))
        return wins

    # 1. Pre-fill balanced buffer before touching weights
    pre_wins = collect_rollouts(160, seed_base=30000)
    print(f"[Pre-fill] collected {len(episodes)} eps | rollout solve: {pre_wins}/160")

    lrs = (6.0e-5, 4.0e-5, 2.5e-5, 1.5e-5)
    for r, lr in enumerate(lrs):
        for pg in opt.param_groups:
            pg["lr"] = lr

        if r > 0:
            wins = collect_rollouts(48, seed_base=31000 + r * 500)
            if len(episodes) > 320:
                episodes = episodes[-320:]
        else:
            wins = pre_wins

        model.policy.train()
        order = np.arange(len(episodes))
        for _ in range(4):
            rng.shuffle(order)
            ls = 0.0
            for idx in order:
                o_np, a_np = episodes[idx]
                starts = torch.zeros(len(o_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)
                dist, _ = model.policy.get_distribution(torch.as_tensor(o_np, device=device), (h0, c0), starts)
                loss = F.smooth_l1_loss(dist.distribution.mean, torch.as_tensor(a_np, device=device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ls += float(loss.item())

        vw, vz, vn = eval_unified(model, seeds=val_seeds, verbose=False)
        tag = ""
        if (vw, vz) >= (best_w, best_z):
            best_w, best_z = vw, vz
            model.save(save_path)
            tag = " <-- NEW BEST"

        denom = 160 if r == 0 else 48
        print(
            f"[Polish R{r} lr={lr:.1e}] rollout: {wins:3d}/{denom} | buf: {len(episodes)} "
            f"| L1: {ls/len(episodes):.4f} | unseen val ({vn} maps): {vw:2d}/{vn} (0-hit: {vz:2d}){tag}"
        )

    best_model = RecurrentPPO.load(save_path)
    print(f"\n=== 80-MAP STRESS TEST (20 Unseen Maps/Family) | {save_path} ===")
    eval_unified(best_model, seeds=range(9800, 9820), verbose=True)


if __name__ == "__main__":
    if "--check-teacher" in sys.argv:
        check_teacher()
    elif "--dagger" in sys.argv:
        train_stage_a_dagger(DAGGER_CKPT)
    elif "--polish" in sys.argv:
        train_focused_polish()
    elif "--ppo" in sys.argv:
        train_stage_b_ppo(DAGGER_CKPT, PPO_CKPT)
    elif "--eval" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--eval"]
        ckpt = args[0] if args else (PPO_CKPT if os.path.exists(PPO_CKPT) else DAGGER_CKPT)
        eval_unified(RecurrentPPO.load(ckpt), seeds=range(9800, 9810), verbose=True)
    elif "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        mtype = args[0] if len(args) > 0 else "hybrid_arena"
        seed = int(args[1]) if len(args) > 1 else 9800
        ckpt = args[2] if len(args) > 2 else (PPO_CKPT if os.path.exists(PPO_CKPT) else DAGGER_CKPT)
        run_viewer(ckpt, map_type=mtype, seed=seed)
    else:
        train_stage_a_dagger(DAGGER_CKPT)
        train_stage_b_ppo(DAGGER_CKPT, PPO_CKPT)
