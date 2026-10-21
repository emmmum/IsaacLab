# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success rate as a function of the ball position: one synchronized episode per env, ball placed on an (x, y) grid.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/heatmap_scoop.py --zero --out <file.npz> [env.<field>=<value> ...]
  ./isaaclab.sh -p logs/scoop_traj_search/heatmap_scoop.py --checkpoint <model.pt> --out <file.npz> [overrides]
"""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Scoop-RL-Direct-v0")
p.add_argument("--checkpoint", default="")
p.add_argument("--zero", action="store_true", help="zero residual = the open-loop trajectory")
p.add_argument("--x", default="0.85,1.15,16", help="ball distance grid: start,stop,count [m]")
p.add_argument("--y", default="-0.04,0.04,9", help="ball lateral grid: start,stop,count [m]")
p.add_argument("--reps", type=int, default=28, help="episodes per grid cell")
p.add_argument("--out", required=True)
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, a):
        import importlib.metadata as metadata

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        xs = np.linspace(*[float(v) for v in a.x.split(",")[:2]], int(a.x.split(",")[2]))
        ys = np.linspace(*[float(v) for v in a.y.split(",")[:2]], int(a.y.split(",")[2]))
        cells = len(xs) * len(ys)
        env_cfg.scene.num_envs = cells * a.reps
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=env_cfg))
        u = env.unwrapped
        dev = u.device
        if a.zero:
            def policy(obs):
                return torch.zeros(env.num_envs, env.num_actions, device=dev)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(a.checkpoint)
            policy = runner.get_inference_policy(device=dev)
        # env e -> cell e % cells
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cell = torch.arange(u.num_envs, device=dev) % cells
        bx = torch.tensor(gx.flatten(), dtype=torch.float32, device=dev)[cell]
        by = torch.tensor(gy.flatten(), dtype=torch.float32, device=dev)[cell]
        u.reset()
        u.place_ball(torch.arange(u.num_envs, device=dev), bx, by)
        obs = env.get_observations()
        success = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
        finished = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
        with torch.no_grad():
            for _ in range(int(u.max_episode_length)):
                obs, rew, dones, _ = env.step(policy(obs))
                success |= (rew > 0) & ~finished
                finished |= dones.bool()
        rate = torch.zeros(cells, device=dev).index_add_(0, cell, success.float()) / a.reps
        rate = rate.reshape(len(xs), len(ys)).cpu().numpy()
        np.savez(a.out, xs=xs, ys=ys, rate=rate, reps=a.reps)
        print(f"HEATMAP {a.out}: mean {100 * rate.mean():.1f}%")
        env.close()


if __name__ == "__main__":
    main()
