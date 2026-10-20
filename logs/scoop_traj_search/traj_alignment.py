# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sensitivity of the robust open-loop scoop trajectory to alignment error (ball distance, lateral offset, heading).

All realistic uncertainty is on (cfg slip defaults + the knobs passed on the command line). Each test point runs
--reps randomized episodes; success = ball in net (contact, off ground, near pocket) for the last 1 s of a 4 s window.
"""

import argparse
import json
import math
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Direct-v0")
p.add_argument("--reps", type=int, default=200)
p.add_argument("--traj", default="best_trajectory_robust.json")
p.add_argument("--out", default="alignment.json")
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, a.traj)))["trajectory"]
BALL_X = 1.0
# (kind, value): dx [m] along the heading, dy [m] lateral, yaw [deg] initial heading error
POINTS = (
    [("dx", v) for v in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)]
    + [("dy", v) for v in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)]
    + [("yaw", v) for v in (2.0, 4.0, 6.0, 8.0, 10.0)]
)
WINDOW, HOLD = 4.0, 1.0


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, a):
        n = len(POINTS) * a.reps
        env_cfg.scene.num_envs = n
        env_cfg.episode_length_s = WINDOW + 2.0
        env_cfg.ball_spawn_x_range = (BALL_X, BALL_X)
        env_cfg.ball_spawn_y_range = (0.0, 0.0)
        env_cfg.arm_locked = False
        env_cfg.arm_monotonic = True
        env_cfg.base_forward_only = True
        env = gym.make(a.task, cfg=env_cfg).unwrapped
        env.reset()
        dev, c = env.device, env.cfg
        o = env.scene.env_origins

        # per-env alignment error: ball position relative to the robot, and robot yaw
        kind = [pt[0] for pt in POINTS] * a.reps
        val = torch.tensor([pt[1] for pt in POINTS] * a.reps, device=dev)
        dx = torch.tensor([v if k == "dx" else 0.0 for k, v in zip(kind, val.tolist())], device=dev)
        dy = torch.tensor([v if k == "dy" else 0.0 for k, v in zip(kind, val.tolist())], device=dev)
        yaw = torch.tensor([math.radians(v) if k == "yaw" else 0.0 for k, v in zip(kind, val.tolist())], device=dev)
        ball_pose = env.ball.data.root_pose_w.torch.clone()
        ball_pose[:, 0] = o[:, 0] + BALL_X + dx
        ball_pose[:, 1] = o[:, 1] + dy
        ball_pose[:, 2] = o[:, 2] + c.ball_radius
        env.ball.write_root_pose_to_sim_index(root_pose=ball_pose)
        env._ball_spawn_pos[:] = ball_pose[:, :3] - o
        root_pose = env.robot.data.root_pose_w.torch.clone()
        root_pose[:, 3:7] = torch.stack(
            [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1
        )  # (x, y, z, w)
        env.robot.write_root_pose_to_sim_index(root_pose=root_pose)

        default = env._default_arm_joint_pos
        dmax = torch.tensor([[T["d_shoulder"], T["d_elbow"]]], device=dev)
        press = T.get("press_elbow", 0.0)
        steps = int(round(WINDOW / env.step_dt))
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        streak = torch.zeros(n, device=dev)
        for k in range(steps):
            t = k * env.step_dt
            act = torch.zeros(n, 5, device=dev)
            act[:, 0] = (2.0 * T["v"] / c.base_lin_vel_scale - 1.0) if t < T["t_lift"] + T["t_brake"] else -1.0
            if t >= T["t_lift"]:
                remaining = (dmax - (env._arm_target - default)).clamp(min=0.0)
                act[:, 3:5] = (remaining.clamp(max=T["rate"] * env.step_dt) / c.arm_increment_max).clamp(0.0, 1.0)
            env._arm_ff_extra[:, 1] = press if t < T["t_lift"] else 0.0
            _, _, term, trunc, _ = env.step(act)
            alive &= ~(term | trunc)
            innet = env._ball_in_net & alive
            streak = torch.where(innet, streak + 1, torch.zeros_like(streak))
        ok = (streak >= int(round(HOLD / env.step_dt))) & alive
        res = []
        okv = ok.view(a.reps, len(POINTS)).float().mean(0)
        for i, (k_, v_) in enumerate(POINTS):
            res.append(dict(kind=k_, value=v_, success=okv[i].item()))
        json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=1)
        print("=" * 60)
        for r in [dict(kind="dx", value=0.0, success=okv[POINTS.index(("dx", 0.0))].item())] + res:
            unit = {"dx": "m", "dy": "m", "yaw": "deg"}[r["kind"]]
            print(f"ALIGN {r['kind']:>3} = {r['value']:+6.2f} {unit}: success {r['success'] * 100:5.1f}%")
        print("=" * 60)
        env.close()


if __name__ == "__main__":
    main()
