import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.hier_macro_maze_env import RealWorldMacroMazeEnv
from env.hier_maze_terrain_env import is_dir_open


def rhr_macro_teacher(env: RealWorldMacroMazeEnv) -> int:
    """Right-Hand-Rule macro teacher from current cell & heading (100% locally observable)."""
    h_walls = env.track["h_walls"]
    v_walls = env.track["v_walls"]
    N = env.track["grid_size"]
    cs = env.track["cell_size"]

    bx, by = float(env.data.qpos[0]), float(env.data.qpos[1])
    w, x, y, z = env.data.qpos[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    r = int(np.clip(round(bx / cs), 0, N - 1))
    c = int(np.clip(round(by / cs), 0, N - 1))
    cur_dir = int(round(yaw / (0.5 * math.pi))) % 4

    # Priority: Right (1), Forward (0), Left (2), Back/U-Turn (3)
    for d_cand, act_id in (
        ((cur_dir - 1) % 4, 1),
        (cur_dir, 0),
        ((cur_dir + 1) % 4, 2),
        ((cur_dir + 2) % 4, 3),
    ):
        if is_dir_open(h_walls, v_walls, r, c, d_cand):
            return act_id
    return 3


def eval_7x7_suite(
    model: RecurrentPPO,
    seeds=range(9600, 9610),
    terrains=("pavement", "sand", "gravel", "ice", "mud"),
    verbose: bool = True,
):
    model.policy.eval()
    tot_w, tot_zh, tot_n = 0, 0, 0
    for tr in terrains:
        env = RealWorldMacroMazeEnv(grid_size=7, terrain=tr, cell_size=0.44, noise_enabled=True)
        w, zh, steps_l, hits_l = 0, 0, [], []
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
            steps_l.append(env.steps)
            hits_l.append(info["wall_hits"])
        n = len(seeds)
        tot_w += w
        tot_zh += zh
        tot_n += n
        if verbose:
            print(
                f"7x7_branch_{tr:9s} | solve: {w:2d}/{n:2d} (0-hit: {zh:2d}) "
                f"| avg cells: {np.mean(steps_l):5.1f} | avg hits: {np.mean(hits_l):4.1f}"
            )
    if verbose:
        print(
            f"\nOVERALL UNSEEN 7x7 BRANCHING TERRAIN SCORE: "
            f"{tot_w}/{tot_n} ({100.0*tot_w/tot_n:.1f}%) | 0-hit: {tot_zh}/{tot_n} ({100.0*tot_zh/tot_n:.1f}%)"
        )
    return tot_w, tot_zh, tot_n


def train_one_shot(save_path: str = "checkpoints/hier_maze_terrain_95pct.zip"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = RealWorldMacroMazeEnv(
        grid_size=5,
        terrain="domain_rand",
        cell_size=(0.40, 0.46),
        base_seed=8100,
        noise_enabled=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[64], vf=[64])),
        verbose=0,
    )

    rng = np.random.default_rng(2026)
    device = model.device
    h_dim = model.policy.lstm_actor.hidden_size
    n_layers = model.policy.lstm_actor.num_layers
    params = [
        p for m in (model.policy.lstm_actor, model.policy.mlp_extractor.policy_net, model.policy.action_net)
        if m is not None for p in m.parameters()
    ]
    opt = torch.optim.Adam(params, lr=1.5e-3)

    betas = (1.0, 0.50, 0.15, 0.00)
    eps_per_round = 36
    epochs_per_round = 16
    lr_start, lr_end = 1.5e-3, 2.0e-4

    episodes = []
    best_score = (-1, -1)

    for r, beta in enumerate(betas):
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * r / (len(betas) - 1)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        wins = 0
        model.policy.eval()
        for ep in range(eps_per_round):
            obs, _ = env.reset(seed=8100 + r * 300 + ep)
            ep_o, ep_a = [], []
            st, ep_s, done = None, np.ones((1,), dtype=bool), False
            while not done:
                ea = rhr_macro_teacher(env)
                sa, st = model.predict(obs, state=st, episode_start=ep_s, deterministic=True)
                ep_s = np.zeros((1,), dtype=bool)
                ep_o.append(obs)
                ep_a.append(ea)
                obs, _, tm, tc, info = env.step(ea if rng.random() < beta else int(sa))
                done = tm or tc
            wins += int(info["is_success"])
            if len(ep_o) > 4:
                episodes.append((np.asarray(ep_o, np.float32), np.asarray(ep_a, np.int64)))

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
                loss = F.cross_entropy(dist.distribution.logits, torch.as_tensor(a_np, device=device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ls += float(loss.item())

        vw, vz, vn = eval_7x7_suite(model, seeds=range(9500, 9504), verbose=False)
        tag = ""
        if (vw, vz) >= best_score:
            best_score = (vw, vz)
            model.save(save_path)
            tag = " <-- NEW BEST"

        print(
            f"[Macro DAgger R{r} beta={beta:.2f} lr={lr:.1e}] rollout: {wins:2d}/{eps_per_round} "
            f"| CE: {ls/len(episodes):.4f} | unseen 7x7 val: {vw:2d}/{vn} (0-hit: {vz:2d}){tag}"
        )

    best_model = RecurrentPPO.load(save_path)
    print(f"\n=== FINAL HELD-OUT 7x7 BRANCHING TERRAIN SUITE (50 Runs, NOISE ON) | {save_path} ===")
    eval_7x7_suite(best_model, seeds=range(9600, 9610), verbose=True)


def run_viewer(ckpt: str, terrain: str = "ice", seed: int = 9600):
    model = RecurrentPPO.load(ckpt)
    env = RealWorldMacroMazeEnv(grid_size=7, terrain=terrain, cell_size=0.44, noise_enabled=True, render_realtime=True)
    obs, _ = env.reset(seed=seed)
    c_mid = 3.0 * 0.44
    print(f"LIVE 7x7 BRANCHING TERRAIN MAZE | terrain={terrain} | seed={seed}")
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
                print(f"{status} | macro_steps={env.steps} | visited={info['visited_cells']}/49 | wall_hits={info['wall_hits']}")
                break


if __name__ == "__main__":
    ckpt = "checkpoints/hier_maze_terrain_95pct.zip"
    if "--check-teacher" in sys.argv:
        for tr in ("pavement", "sand", "gravel", "ice", "mud"):
            env = RealWorldMacroMazeEnv(grid_size=7, terrain=tr, cell_size=0.44, noise_enabled=True)
            w, zh = 0, 0
            for s in range(9600, 9605):
                env.reset(seed=s)
                done = False
                while not done:
                    _, _, tm, tc, info = env.step(rhr_macro_teacher(env))
                    done = tm or tc
                w += int(info["is_success"])
                zh += int(info["is_success"] and info["wall_hits"] == 0)
            print(f"Macro Teacher 7x7 {tr:9s}: {w}/5 (0-hit: {zh}/5)")
    elif "--view" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--view"]
        terrain = args[0] if len(args) > 0 else "ice"
        seed = int(args[1]) if len(args) > 1 else 9600
        run_viewer(ckpt, terrain=terrain, seed=seed)
    elif "--eval" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--eval"]
        eval_7x7_suite(RecurrentPPO.load(args[0] if args else ckpt), seeds=range(9600, 9610), verbose=True)
    else:
        train_one_shot(ckpt)
