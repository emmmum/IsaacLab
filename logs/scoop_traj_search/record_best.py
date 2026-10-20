# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play the best open-loop scoop trajectory in one env, record states and rendered frames.

Usage: ./isaaclab.sh -p logs/scoop_traj_search/record_best.py --enable_cameras --out_dir <dir>
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Direct-v0")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--window", type=float, default=3.0)
parser.add_argument("--max_tries", type=int, default=5)
parser.add_argument("--render", action="store_true", help="also capture RTX frames (needs --enable_cameras)")
parser.add_argument("--traj", default="best_trajectory.json", help="trajectory json in this folder")
parser.add_argument("--old_pd", action="store_true", help="use the arm PD the trajectory was tuned for (Kp 200, Kd 20, 50 N m)")
add_launcher_args(parser)
args, rem = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + rem

TRAJ_FILE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.traj)))
TRAJ = TRAJ_FILE["trajectory"]
# the old-PD trajectory was tuned with the elbow starting at -0.632 rad
ELBOW_DEFAULT = TRAJ_FILE["env_overrides"].get("robot_cfg.init_state.joint_pos.RevoluteJoint2")
PRESS = TRAJ.get("press_elbow", 0.0)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = args.window + 2.0
        if ELBOW_DEFAULT is not None:
            env_cfg.robot_cfg.init_state.joint_pos["RevoluteJoint2"] = ELBOW_DEFAULT
        env_cfg.base_slip_scale_range = (1.0, 1.0)
        env_cfg.base_slip_bias_max = (0.0, 0.0, 0.0)
        env_cfg.base_slip_noise_std = (0.0, 0.0, 0.0)
        env_cfg.base_slip_lag_tau_range = (0.1, 0.1)
        env_cfg.ball_spawn_x_range = (1.0, 1.0)
        env_cfg.ball_spawn_y_range = (0.0, 0.0)
        if args.old_pd:
            arm = env_cfg.robot_cfg.actuators["arm"]
            arm.stiffness, arm.damping, arm.effort_limit_sim = 200.0, 20.0, 50.0
        env_cfg.arm_locked = False
        env_cfg.arm_monotonic = True
        env_cfg.base_forward_only = True
        # side view from -y, following the action along +x
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (0.85, -2.1, 0.55)
        env_cfg.viewer.lookat = (0.85, 0.0, 0.3)
        env_cfg.viewer.resolution = (1280, 720)
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.render else None).unwrapped
        c, dev = env.cfg, env.device
        v, tl, rate = TRAJ["v"], TRAJ["t_lift"], TRAJ["rate"]
        dmax_lift = torch.tensor([[TRAJ["d_shoulder"], TRAJ["d_elbow"]]], device=dev)
        t_stop = tl + TRAJ["t_brake"]
        steps = int(round(args.window / env.step_dt))
        elbow_off = torch.tensor([[0.0, 0.016, 0.48]], device=dev)  # elbow joint in the lowerarm frame
        from isaaclab.utils.math import quat_apply

        for attempt in range(args.max_tries):
            env.reset()
            # warm up the renderer so the first frames are not blank
            for _ in range(3 if args.render else 0):
                env.sim.render()
            default = env._default_arm_joint_pos
            rec = {k: [] for k in ["t", "vx_cmd", "vx", "sh_target", "el_target", "sh", "el", "base_x", "base_z", "shoulder_x",
                                  "shoulder_z", "elbow_x", "elbow_z", "pocket_x", "pocket_z", "lip_x", "lip_z", "ball_x",
                                  "ball_z", "in_net", "net_force", "sh_torque", "el_torque"]}
            frames = []
            in_net_streak, max_streak = 0, 0
            for k in range(steps):
                t = (k + 1) * env.step_dt
                act = torch.zeros(1, 5, device=dev)
                vx_cmd = v if k * env.step_dt < t_stop else 0.0
                act[:, 0] = 2.0 * vx_cmd / c.base_lin_vel_scale - 1.0
                if k * env.step_dt >= tl:
                    remaining = (dmax_lift - (env._arm_target - default)).clamp(min=0.0)
                    inc = torch.clamp(remaining, max=rate * env.step_dt)
                    act[:, 3:5] = (inc / c.arm_increment_max).clamp(0.0, 1.0)
                env._arm_ff_extra[:, 1] = PRESS if k * env.step_dt < tl else 0.0
                env.step(act)
                if args.render:
                    frames.append(env.render())
                o = env.scene.env_origins[0]
                rp = env.robot.data.root_pos_w.torch[0] - o
                sh_p = env.robot.data.body_pos_w.torch[0, env._shoulder_body_idx] - o  # shoulder joint
                la_p = env.robot.data.body_pos_w.torch[0, env._lowerarm_body_idx]
                la_q = env.robot.data.body_quat_w.torch[0, env._lowerarm_body_idx]
                elbow = (la_p + quat_apply(la_q[None], elbow_off)[0]) - o
                pk = env._pocket_pos_w()[0] - o
                lip = env._lip_pos_w()[0] - o
                ball = env.ball.data.root_pos_w.torch[0] - o
                j = env.robot.data.joint_pos.torch[0, env._arm_joint_idx]
                f = torch.linalg.norm(env.ball_contact.data.force_matrix_w_history.torch[0], dim=-1).max()
                tq = env.robot.data.applied_torque.torch[0, env._arm_joint_idx]
                innet = bool(env._ball_in_net[0])
                in_net_streak = in_net_streak + 1 if innet else 0
                max_streak = max(max_streak, in_net_streak)
                vals = dict(t=t, vx_cmd=vx_cmd, vx=env.robot.data.root_lin_vel_b.torch[0, 0].item(),
                            sh_target=env._arm_target[0, 0].item(), el_target=env._arm_target[0, 1].item(),
                            sh=j[0].item(), el=j[1].item(), base_x=rp[0].item(), base_z=rp[2].item(),
                            shoulder_x=sh_p[0].item(), shoulder_z=sh_p[2].item(), elbow_x=elbow[0].item(), elbow_z=elbow[2].item(),
                            pocket_x=pk[0].item(), pocket_z=pk[2].item(), lip_x=lip[0].item(), lip_z=lip[2].item(),
                            ball_x=ball[0].item(), ball_z=ball[2].item(), in_net=float(innet), net_force=f.item(), sh_torque=tq[0].item(), el_torque=tq[1].item())
                for kk, vv in vals.items():
                    rec[kk].append(vv)
            success = in_net_streak >= int(1.0 / env.step_dt)
            print(f"RECORD attempt {attempt}: success={success} final in-net streak {in_net_streak} steps")
            if success:
                break
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez(os.path.join(args.out_dir, "trajectory_record.npz"), **{k: np.array(vv) for k, vv in rec.items()})
        if frames:
            import imageio

            # sim steps are 60 Hz: write at 60 fps (real time) and 15 fps (4x slow motion)
            rgb = [np.asarray(f)[..., :3] for f in frames]
            imageio.mimwrite(os.path.join(args.out_dir, "render_realtime.mp4"), rgb, fps=60, quality=8)
            imageio.mimwrite(os.path.join(args.out_dir, "render_slowmo_4x.mp4"), rgb, fps=15, quality=8)
        print(f"RECORD saved {len(rec['t'])} steps, frames {len(frames)}, success={success}")
        env.close()


if __name__ == "__main__":
    main()
