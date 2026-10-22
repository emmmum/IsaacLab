# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the stick's world motion during the RoboMaster scoop (for retargeting to another arm).

Runs the open-loop trajectory (zero residual) on the flat floor with the ball centered on the stick head and saves,
at every physics step, the stick body (``lowerarm``) pose and velocity, the lip and pocket positions, the ball pose
and the ball-stick contact flag, all relative to the env origin. Several envs are recorded so the spread can be seen.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/record_stick_traj.py --viz none --out stick_traj.npz [--num_envs 16]
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
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--out", required=True)
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, a):
        env_cfg.scene.num_envs = a.num_envs
        env_cfg.ball_spawn_x_range = (1.0, 1.0)
        env_cfg.ball_spawn_y_range = (0.0, 0.0)
        u = gym.make(a.task, cfg=env_cfg).unwrapped
        u.reset()
        o = u.scene.env_origins
        b = u._lowerarm_body_idx
        rec = {k: [] for k in ("stick_pos", "stick_quat", "stick_lin_vel", "stick_ang_vel", "lip", "pocket", "ball", "contact")}
        orig_update = u.scene.update

        def update_and_record(*args, **kwargs):
            orig_update(*args, **kwargs)
            d = u.robot.data
            rec["stick_pos"].append(d.body_pos_w.torch[:, b] - o)
            rec["stick_quat"].append(d.body_quat_w.torch[:, b].clone())  # (x, y, z, w)
            rec["stick_lin_vel"].append(d.body_lin_vel_w.torch[:, b].clone())
            rec["stick_ang_vel"].append(d.body_ang_vel_w.torch[:, b].clone())
            rec["lip"].append(u._lip_pos_w() - o)
            rec["pocket"].append(u._pocket_pos_w() - o)
            rec["ball"].append(u.ball.data.root_pos_w.torch - o)
            f = u.ball_contact.data.force_matrix_w.torch
            rec["contact"].append(torch.linalg.norm(f, dim=-1).amax(dim=(1, 2)) > u.cfg.net_contact_force_threshold)

        u.scene.update = update_and_record
        succ = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        with torch.no_grad():
            for _ in range(int(u.max_episode_length) - 1):
                _, r, _, _, _ = u.step(torch.zeros(u.num_envs, 3, device=u.device))
                succ |= r > 0
        out = {k: torch.stack(v, dim=1).cpu().numpy() for k, v in rec.items()}  # (num_envs, T, ...)
        np.savez(a.out, physics_dt=u.physics_dt, success=succ.cpu().numpy(), **out)
        print(f"RECORD {a.out}: {out['stick_pos'].shape[1]} physics steps, success {int(succ.sum())}/{u.num_envs}")
        u.close()


if __name__ == "__main__":
    main()
