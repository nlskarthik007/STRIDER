import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.hier_continuous_euclidean_maze_env import RealWorldContinuousEuclideanMazeEnv
from env.hier_maze_terrain_env import DIRS, is_dir_open


class ContinuousEuclideanTeacher:
    """
    Continuous-motion Euclidean Best-First DFS teacher:
    - Picks unvisited branches minimizing Euclidean distance to the goal.
    - Backtracks cleanly out of dead ends via DFS stack.
    - Glides non-stop through straightaways and carves smooth continuous 90-deg arcs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.visited = set()
        self.stack = []
        self.last_cell = None
        self.in_dir = 0
        self.target_dir = 0
        self.target_cell = (0, 0)
        self.reached_center = False

    def _choose_next(self, h_walls, v_walls, r: int, c: int, N: int):
        goal_r, goal_c = N - 1, N - 1
        unvisited = []
        for d, (dr, dc) in enumerate(DIRS):
            if is_dir_open(h_walls, v_walls, r, c, d):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in self.visited:
                    dist_sq = (nr - goal_r) ** 2 + (nc - goal_c) ** 2
                    unvisited.append((dist_sq, d, (nr, nc)))

        if unvisited:
            unvisited.sort(key=lambda x: x[0])
            _, best_d, best_cell = unvisited[0]
            return best_d, best_cell

        # Dead end or exhausted junction -> backtrack along DFS stack
        if len(self.stack) >= 2:
            pr, pc = self.stack[-2]
            dr, dc = pr - r, pc - c
            if (dr, dc) in DIRS:
                return DIRS.index((dr, dc)), (pr, pc)

        # Fallback: open neighbor closest in Euclidean distance
        cands = []
        for d, (dr, dc) in enumerate(DIRS):
            if is_dir_open(h_walls, v_walls, r, c, d):
                nr, nc = r + dr, c + dc
                cands.append(((nr - goal_r) ** 2 + (nc - goal_c) ** 2, d, (nr, nc)))
        cands.sort(key=lambda x: x[0])
        return cands[0][1], cands[0][2]

    def act(self, env: RealWorldContinuousEuclideanMazeEnv) -> np.ndarray:
        h_walls = env.track["h_walls"]
        v_walls = env.track["v_walls"]
        N = env.track["grid_size"]
        cs = env.track["cell_size"]

        bx, by, yaw = env._body_pose()
        r = int(np.clip(round(bx / cs), 0, N - 1))
        c = int(np.clip(round(by / cs), 0, N - 1))
        cell = (r, c)

        if self.last_cell is None:
            self.visited = {cell}
            self.stack = [cell]
            self.in_dir = env.track["init_dir"]
            self.target_dir, self.target_cell = self._choose_next(h_walls, v_walls, r, c, N)
            self.last_cell = cell
            self.reached_center = True
        elif cell != self.last_cell:
            dr = cell[0] - self.last_cell[0]
            dc = cell[1] - self.last_cell[1]
            if (dr, dc) in DIRS:
                self.in_dir = DIRS.index((dr, dc))
            else:
                self.in_dir = int(round(yaw / (0.5 * math.pi))) % 4

            if cell in self.stack:
                self.stack = self.stack[: self.stack.index(cell) + 1]
            else:
                self.stack.append(cell)
            self.visited.add(cell)

            self.target_dir, self.target_cell = self._choose_next(h_walls, v_walls, r, c, N)
            self.last_cell = cell
            # Stay continuous (no pause) when continuing straight
            self.reached_center = (self.target_dir == self.in_dir)

        cx, cy = r * cs, c * cs
        in_dx, in_dy = DIRS[self.in_dir]
        along = (cx - bx) * in_dx + (cy - by) * in_dy
        rf_f = env._read_rf("rf_front")
        rf_l = env._read_rf("rf_left")
        rf_r = env._read_rf("rf_right")

        # Approach cell center before turning so we never clip inner corners
        if (not self.reached_center) and (along <= 0.085 or rf_f < 0.16):
            self.reached_center = True

        if not self.reached_center:
            tx, ty = cx - bx, cy - by
            v_cap = 0.28
        else:
            tx = self.target_cell[0] * cs - bx
            ty = self.target_cell[1] * cs - by
            # Look one cell further when straightaway continues for full continuous speed
            tr, tc = self.target_cell
            straight_ahead = (
                self.target_dir == self.in_dir
                and is_dir_open(h_walls, v_walls, tr, tc, self.target_dir)
                and (tr + DIRS[self.target_dir][0], tc + DIRS[self.target_dir][1]) not in self.visited
            )
            v_cap = 0.44 if straight_ahead else 0.30

        desired_yaw = math.atan2(ty, tx)
        alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Dead-end 180-deg reversal: deterministic CCW Gyro-PI pivot (zero ±pi flip)
        if abs(alpha) > 2.15:
            v_des = 0.0
            w_des = 1.95
        else:
            # Smooth continuous arc law: full speed on straights, smooth curve around 90-deg corners
            v_des = v_cap * max(0.0, math.cos(alpha)) ** 1.4
            if abs(alpha) > 1.15:
                v_des = min(v_des, 0.06)
            w_des = 2.60 * alpha

            # Subtle corridor centering when aligned in a straight passage
            if abs(alpha) < 0.35 and min(rf_l, rf_r) < 0.15:
                w_des += 1.2 * float(np.clip((rf_l - rf_r) / 0.14, -1.0, 1.0))

        v_norm = float(np.clip(2.0 * (v_des / 0.48) - 1.0, -1.0, 1.0))
        w_norm = float(np.clip(w_des / 2.20, -1.0, 1.0))
        return np.array([v_norm, w_norm], dtype=np.float32)


def eval_7x7_continuous(
    model: RecurrentPPO,
    seeds=range(9600, 9610),
    terrains=("pavement", "sand", "gravel", "ice", "mud"),
    verbose: bool = True,
):
    model.policy.eval()
    tot_w, tot_zh, tot_n = 0, 0, 0
    for tr in terrains:
        env = RealWorldContinuousEuclideanMazeEnv(grid_size=7, terrain=tr, cell_size=0.44, noise_enabled=True)
        w, zh, steps_l, hits_l = 0, 0, [], []
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
            steps_l.append(env.steps)
            hits_l.append(info["wall_hits"])
        n = len(seeds)
        tot_w += w
        tot_zh += zh
        tot_n += n
        if verbose:
            print(
                f"7x7_cont_euc_{tr:9s} | solve: {w:2d}/{n:2d} (0-hit: {zh:2d}) "
                f"| avg time: {np.mean(steps_l)*0.1:5.1f}s | avg hits: {np.mean(hits_l):4.1f}"
            )
    if verbose:
        print(
            f"\nOVERALL CONTINUOUS EUCLIDEAN 7x7 SCORE: "
            f"{tot_w}/{tot_n} ({100.0*tot_w/tot_n:.1f}%) | 0-hit: {tot_zh}/{tot_n}"
        )
    return tot_w, tot_zh, tot_n


def train_continuous_dagger(save_path: str = "checkpoints/hier_cont_euclidean_maze.zip"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = RealWorldContinuousEuclideanMazeEnv(
        grid_size=(5, 6),
        terrain="domain_rand",
        cell_size=(0.40, 0.46),
        base_seed=9100,
        noise_enabled=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[64], vf=[64])),
        verbose=0,
    )

    teacher = ContinuousEuclideanTeacher()
    rng = np.random.default_rng(2026)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers
    params = [
        p for m in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net)
        if m is not None for p in m.parameters()
    ]
    opt = torch.optim.Adam(params, lr=1.2e-3)

    betas = (1.0, 0.65, 0.35, 0.15, 0.05, 0.00)
    eps_per_round = 36
    epochs_per_round = 14
    lr_start, lr_end = 1.2e-3, 1.5e-4

    episodes = []
    best_score = (-1, -1)

    for r, beta in enumerate(betas):
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * r / (len(betas) - 1)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=9100 + r * 300 + ep)
            teacher.reset()
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
            if len(ep_o) > 8:
                episodes.append((np.asarray(ep_o, np.float32), np.asarray(ep_a, np.float32)))

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

        vw, vz, vn = eval_7x7_continuous(model, seeds=range(9500, 9504), verbose=False)
        tag = ""
        if (vw, vz) >= best_score:
            best_score = (vw, vz)
            model.save(save_path)
            tag = " <-- NEW BEST"

        print(
            f"[Cont-Euc DAgger R{r} beta={beta:.2f} lr={lr:.1e}] rollout: {wins:2d}/{eps_per_round} "
            f"| L1: {ls/len(episodes):.4f} | unseen 7x7 val: {vw:2d}/{vn} (0-hit: {vz:2d}){tag}"
        )

    best_model = RecurrentPPO.load(save_path)
    print(f"\n=== FINAL HELD-OUT CONTINUOUS EUCLIDEAN 7x7 SUITE (50 Runs) | {save_path} ===")
    eval_7x7_continuous(best_model, seeds=range(9600, 9610), verbose=True)


def run_viewer(ckpt: str, terrain: str = "ice", seed: int = 9600, grid_size: int = 7):
    model = RecurrentPPO.load(ckpt)
    env = RealWorldContinuousEuclideanMazeEnv(
        grid_size=grid_size, terrain=terrain, cell_size=0.44, noise_enabled=True, render_realtime=True
    )
    obs, _ = env.reset(seed=seed)
    c_mid = 0.5 * (grid_size - 1) * 0.44
    print(f"LIVE CONTINUOUS EUCLIDEAN {grid_size}x{grid_size} | terrain={terrain} | seed={seed}")
    st, ep = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [c_mid, c_mid, 0.0]
        viewer.cam.distance = 4.4 if grid_size <= 7 else 5.6
        viewer.cam.elevation = -72.0
        while viewer.is_running():
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, info = env.step(a)
            if tm or tc:
                status = "GOAL REACHED" if info["is_success"] else "ENDED"
                print(
                    f"{status} | time={env.steps*0.1:.1f}s | visited={info['visited_cells']}/{grid_size*grid_size} "
                    f"| wall_hits={info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    ckpt = "checkpoints/hier_cont_euclidean_maze.zip"
    if "--check-teacher" in sys.argv:
        teacher = ContinuousEuclideanTeacher()
        for tr in ("pavement", "sand", "gravel", "ice", "mud"):
            env = RealWorldContinuousEuclideanMazeEnv(grid_size=7, terrain=tr, cell_size=0.44, noise_enabled=True)
            w, zh, t_l = 0, 0, []
            for s in range(9600, 9605):
                env.reset(seed=s)
                teacher.reset()
                done = False
                while not done:
                    _, _, tm, tc, info = env.step(teacher.act(env))
                    done = tm or tc
                w += int(info["is_success"])
                zh += int(info["is_success"] and info["wall_hits"] == 0)
                t_l.append(env.steps * 0.1)
            print(f"Cont-Euc Teacher 7x7 {tr:9s}: {w}/5 (0-hit: {zh}/5) | avg time: {np.mean(t_l):4.1f}s")
    elif "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        terrain = args[0] if len(args) > 0 else "ice"
        seed = int(args[1]) if len(args) > 1 else 9600
        grid = int(args[2]) if len(args) > 2 else 7
        run_viewer(ckpt, terrain=terrain, seed=seed, grid_size=grid)
    elif "--eval" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--eval"]
        eval_7x7_continuous(RecurrentPPO.load(args[0] if args else ckpt), seeds=range(9600, 9610), verbose=True)
    else:
        train_continuous_dagger(ckpt)
