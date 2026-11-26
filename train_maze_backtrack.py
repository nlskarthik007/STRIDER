import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.hier_maze_terrain_env import (
    RealWorldTerrainMazeEnv,
    DIRS,
    is_dir_open,
)
from train_95plus import centering_pursuit_teacher


class DynamicRHRTeacher:
    """
    Cell-latched Right-Hand-Rule explorer teacher.
    Works from ANY off-policy cell (safe for DAgger beta -> 0) and performs clean
    center-aligned 90-deg turns and 180-deg dead-end reversals using Gyro-PI spins.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_cell = None
        self.in_dir = 0
        self.target_dir = 0
        self.target_cell = (0, 0)
        self.reached_center = False

    def _quantize_yaw(self, yaw: float) -> int:
        # 0: 0 rad (+x), 1: +pi/2 (+y), 2: pi (-x), 3: -pi/2 (-y)
        return int(round(yaw / (0.5 * math.pi))) % 4

    def _choose_rhr(self, h_walls, v_walls, r: int, c: int, in_dir: int) -> int:
        # Strict Right-Hand Rule: Right, Straight, Left, Back (180 deg dead-end reversal)
        for cand in ((in_dir - 1) % 4, in_dir, (in_dir + 1) % 4, (in_dir + 2) % 4):
            if is_dir_open(h_walls, v_walls, r, c, cand):
                return cand
        return (in_dir + 2) % 4

    def act(self, env: RealWorldTerrainMazeEnv) -> int:
        if env.track["track_type"] != "maze":
            return centering_pursuit_teacher(env)

        h_walls = env.track["h_walls"]
        v_walls = env.track["v_walls"]
        N = env.track["grid_size"]
        cs = env.track["cell_size"]

        bx, by = float(env.data.qpos[0]), float(env.data.qpos[1])
        w, x, y, z = env.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        r = int(np.clip(round(bx / cs), 0, N - 1))
        c = int(np.clip(round(by / cs), 0, N - 1))
        cell = (r, c)

        if self.last_cell is None:
            self.in_dir = env.track["init_dir"]
            self.target_dir = self._choose_rhr(h_walls, v_walls, r, c, self.in_dir)
            dr, dc = DIRS[self.target_dir]
            self.target_cell = (r + dr, c + dc)
            self.last_cell = cell
            self.reached_center = True
        elif cell != self.last_cell:
            dr = cell[0] - self.last_cell[0]
            dc = cell[1] - self.last_cell[1]
            if (dr, dc) in DIRS:
                self.in_dir = DIRS.index((dr, dc))
            else:
                self.in_dir = self._quantize_yaw(yaw)
            self.target_dir = self._choose_rhr(h_walls, v_walls, r, c, self.in_dir)
            tdr, tdc = DIRS[self.target_dir]
            self.target_cell = (r + tdr, c + tdc)
            self.last_cell = cell
            self.reached_center = (self.target_dir == self.in_dir)

        cx, cy = r * cs, c * cs
        in_dx, in_dy = DIRS[self.in_dir]
        along = (cx - bx) * in_dx + (cy - by) * in_dy
        rf_f = env._read_rf("rf_front")

        if (not self.reached_center) and (along <= 0.065 or rf_f < 0.14):
            self.reached_center = True

        if not self.reached_center:
            tx, ty = cx - bx, cy - by
        else:
            tx = self.target_cell[0] * cs - bx
            ty = self.target_cell[1] * cs - by

        desired_yaw = math.atan2(ty, tx)
        alpha = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Trigger in-place Gyro-PI pivot for sharp 90-deg turns & 180-deg dead-end U-turns
        if abs(alpha) > 0.95:
            return 5 if alpha > 0.0 else 6

        ld = max(0.14, math.hypot(tx, ty))
        kappa = 2.0 * math.sin(alpha) / ld + 0.72 * alpha

        rf_l = env._read_rf("rf_left")
        rf_r = env._read_rf("rf_right")
        if min(rf_l, rf_r) < 0.11:
            kappa += 0.40 * float(np.clip((rf_l - rf_r) / 0.14, -1.0, 1.0))

        if abs(kappa) < 0.42:
            return 0
        if kappa >= 1.55:
            return 3
        if kappa > 0.0:
            return 1
        if kappa <= -1.55:
            return 4
        return 2


def transfer_5act_to_7act(new_model: RecurrentPPO, src_ckpt: str):
    """Copy all compatible LSTM/MLP weights and slice-transfer the 5 forward-arc rows."""
    base = RecurrentPPO.load(src_ckpt, device="cpu")
    src_sd = base.policy.state_dict()
    dst_sd = new_model.policy.state_dict()

    copied = 0
    for k, v in src_sd.items():
        if k not in dst_sd:
            continue
        if v.shape == dst_sd[k].shape:
            dst_sd[k] = v.to(dst_sd[k].device)
            copied += 1
        elif k == "action_net.weight" and v.shape[0] == 5 and dst_sd[k].shape[0] == 7:
            dst_sd[k][:5] = v.to(dst_sd[k].device)
            dst_sd[k][5:] *= 0.1
            copied += 1
        elif k == "action_net.bias" and v.shape[0] == 5 and dst_sd[k].shape[0] == 7:
            dst_sd[k][:5] = v.to(dst_sd[k].device)
            dst_sd[k][5:] = 0.0
            copied += 1

    new_model.policy.load_state_dict(dst_sd)
    print(f"Transferred {copied} tensors (including 5 forward-arc heads) from {src_ckpt}")


def eval_unseen_7x7_branching(model: RecurrentPPO, seeds=range(9500, 9504), terrains=("pavement", "sand", "gravel", "ice", "mud"), verbose=False):
    model.policy.eval()
    total_wins, total_zh, total_runs = 0, 0, 0
    for tr in terrains:
        env = RealWorldTerrainMazeEnv(mode="maze", grid_size=7, terrain=tr, corridor_width=0.44, noise_enabled=True)
        w, zh, hits_list, steps_list = 0, 0, [], []
        for s in seeds:
            obs, _ = env.reset(seed=s)
            st, ep, done = None, np.ones((1,), dtype=bool), False
            while not done:
                a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
                ep = np.zeros((1,), dtype=bool)
                obs, _, tm, tc, info = env.step(int(a))
                done = tm or tc
            w += int(info["is_success"])
            zh += int(info["is_success"] and info["wall_hits"] == 0)
            hits_list.append(info["wall_hits"])
            steps_list.append(env.steps)
        n = len(seeds)
        total_wins += w
        total_zh += zh
        total_runs += n
        if verbose:
            print(
                f"7x7_branch_{tr:9s} | solve: {w:2d}/{n:2d} (0-hit: {zh:2d}) "
                f"| avg steps: {np.mean(steps_list):6.1f} | avg hits: {np.mean(hits_list):4.1f}"
            )
    if verbose:
        print(f"\nOVERALL UNSEEN 7x7 BRANCHING TERRAIN SCORE: {total_wins}/{total_runs} ({100.0*total_wins/total_runs:.1f}%) | 0-hit: {total_zh}/{total_runs}")
    return total_wins, total_zh, total_runs


def train_backtrack_dagger(src_ckpt: str, save_dir: str = "checkpoints/hier_maze_terrain"):
    os.makedirs(save_dir, exist_ok=True)
    env = RealWorldTerrainMazeEnv(
        mode="mixed",
        grid_size=(5, 6),
        terrain="domain_rand",
        corridor_width=(0.35, 0.46),
        track_type="train_mix",
        base_seed=6100,
        noise_enabled=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3.0e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[64], vf=[64])),
        verbose=0,
    )
    if os.path.exists(src_ckpt):
        transfer_5act_to_7act(model, src_ckpt)

    teacher = DynamicRHRTeacher()
    rng = np.random.default_rng(2026)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers

    actor_params = []
    for mod in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net):
        if mod is not None:
            actor_params.extend(list(mod.parameters()))
    opt = torch.optim.Adam(actor_params, lr=8.0e-4)

    betas = (1.0, 0.65, 0.40, 0.20, 0.08, 0.00)
    eps_per_round = 32
    epochs_per_round = 14
    lr_start, lr_end = 8.0e-4, 1.0e-4

    episodes = []
    best_score = (-1, -1)
    best_path = os.path.join(save_dir, "maze_terrain_best.zip")

    for r, beta in enumerate(betas):
        frac = r / max(1, len(betas) - 1)
        cur_lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=6100 + r * 300 + ep)
            teacher.reset()
            ep_obs, ep_act = [], []
            st, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done:
                exp_act = teacher.act(env)
                stu_act, st = model.predict(obs, state=st, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
                step_act = exp_act if (rng.random() < beta) else int(stu_act)

                ep_obs.append(obs)
                ep_act.append(exp_act)
                obs, _, tm, tc, info = env.step(step_act)
                done = tm or tc

            wins += int(info["is_success"])
            if len(ep_obs) > 8:
                episodes.append((np.asarray(ep_obs, dtype=np.float32), np.asarray(ep_act, dtype=np.int64)))

        model.policy.train()
        order = np.arange(len(episodes))
        for _ in range(epochs_per_round):
            rng.shuffle(order)
            loss_sum = 0.0
            for idx in order:
                obs_np, act_np = episodes[idx]
                obs_t = torch.as_tensor(obs_np, device=device)
                act_t = torch.as_tensor(act_np, device=device)
                starts = torch.zeros(len(obs_np), dtype=torch.float32, device=device)
                starts[0] = 1.0
                h0 = torch.zeros((n_layers, 1, h_dim), device=device)
                c0 = torch.zeros((n_layers, 1, h_dim), device=device)

                dist, _ = model.policy.get_distribution(obs_t, (h0, c0), starts)
                loss = F.cross_entropy(dist.distribution.logits, act_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
                opt.step()
                loss_sum += float(loss.item())

        vw, vz, vn = eval_unseen_7x7_branching(model, seeds=range(9500, 9504))
        tag = ""
        if (vw, vz) >= best_score:
            best_score = (vw, vz)
            model.save(best_path)
            tag = " <-- NEW BEST"

        print(
            f"[DAgger R{r} beta={beta:.2f} lr={cur_lr:.1e}] rollout: {wins:2d}/{eps_per_round} "
            f"| CE: {loss_sum/len(episodes):.4f} | unseen 7x7 val: {vw:2d}/{vn} (0-hit: {vz:2d}){tag}"
        )

    print(f"\nFinished! Best unseen 7x7 model saved to {best_path}")


def run_viewer(ckpt: str, terrain: str = "sand", seed: int = 9600, grid_size: int = 7):
    model = RecurrentPPO.load(ckpt)
    env = RealWorldTerrainMazeEnv(
        mode="maze",
        grid_size=grid_size,
        terrain=terrain,
        corridor_width=0.44,
        noise_enabled=True,
        render_realtime=True,
    )
    obs, _ = env.reset(seed=seed)
    c_mid = 0.5 * (grid_size - 1) * 0.44
    print(f"LIVE {grid_size}x{grid_size} BRANCHING MAZE | terrain={terrain} | seed={seed}")
    st, ep = None, np.ones((1,), dtype=bool)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        viewer.cam.lookat[:] = [c_mid, c_mid, 0.0]
        viewer.cam.distance = 4.4
        viewer.cam.elevation = -72.0
        while viewer.is_running():
            a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True)
            ep = np.zeros((1,), dtype=bool)
            obs, _, tm, tc, info = env.step(int(a))
            if tm or tc:
                status = "GOAL REACHED" if info["is_success"] else "ENDED"
                print(
                    f"{status} | steps={env.steps} | visited={info['visited_cells']}/{grid_size*grid_size} "
                    f"| wall_hits={info['wall_hits']}"
                )
                break


if __name__ == "__main__":
    if "--check-teacher" in sys.argv:
        teacher = DynamicRHRTeacher()
        for tr in ("pavement", "sand", "gravel", "ice", "mud"):
            env = RealWorldTerrainMazeEnv(mode="maze", grid_size=7, terrain=tr, corridor_width=0.44, noise_enabled=True)
            w, zh = 0, 0
            for s in range(9600, 9605):
                obs, _ = env.reset(seed=s)
                teacher.reset()
                done = False
                while not done:
                    _, _, tm, tc, info = env.step(teacher.act(env))
                    done = tm or tc
                w += int(info["is_success"])
                zh += int(info["is_success"] and info["wall_hits"] == 0)
            print(f"Teacher 7x7 branching {tr:9s}: {w}/5 (0-hit: {zh}/5)")
    elif "--eval" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--eval"]
        ckpt = args[0] if args else "checkpoints/hier_maze_terrain/maze_terrain_best.zip"
        model = RecurrentPPO.load(ckpt)
        print(f"\n=== ZERO-SHOT UNSEEN 7x7 BRANCHING TERRAIN MAZES (50 Runs) | {ckpt} ===")
        eval_unseen_7x7_branching(model, seeds=range(9600, 9610), verbose=True)
    elif "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        terrain = args[0] if len(args) > 0 else "sand"
        seed = int(args[1]) if len(args) > 1 else 9600
        ckpt = args[2] if len(args) > 2 else "checkpoints/hier_maze_terrain/maze_terrain_best.zip"
        run_viewer(ckpt, terrain=terrain, seed=seed, grid_size=7)
    else:
        src = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hier_curve_champion_98pct.zip"
        train_backtrack_dagger(src)
