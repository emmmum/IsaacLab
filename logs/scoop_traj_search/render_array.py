# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an N x N array of robots running the robust open-loop scoop trajectory, one frame per physics step.

Physics runs at 1200 Hz with the controller stepped every physics step (decimation 1; the open-loop trajectory is
piecewise constant, so this is equivalent to 60 Hz control). Frames are streamed to an mp4 at --fps, e.g. 240 fps for
5x slow motion. All realistic uncertainty is on (pass the knobs as hydra overrides), plus a small random ball offset
per robot inside the measured alignment tolerance.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/render_array.py --headless --enable_cameras --out <file.mp4> <overrides>
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
p.add_argument("--grid", type=int, default=10)
p.add_argument("--spacing", type=float, default=2.5)
p.add_argument("--window", type=float, default=3.0)
p.add_argument("--fps", type=int, default=240)
p.add_argument("--physics_hz", type=int, default=1200)
p.add_argument("--control_hz", type=int, default=60, help="policy/trajectory rate; frames are still grabbed every physics step")
p.add_argument("--ball_dx", type=float, default=0.05, help="uniform +- ball distance offset [m]")
p.add_argument("--ball_dy", type=float, default=0.015, help="uniform +- ball lateral offset [m]")
p.add_argument("--traj", default="best_trajectory_robust.json")
p.add_argument("--record", default="", help="save per-physics-step robot/ball states of every env to this .npz")
p.add_argument("--out", required=True, help="mp4 path, or 'none' to skip rendering")
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, a.traj)))["trajectory"]
BALL_X = 1.0


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, a):
        n = a.grid * a.grid
        env_cfg.scene.num_envs = n
        env_cfg.scene.env_spacing = a.spacing
        env_cfg.sim.dt = 1.0 / a.physics_hz
        env_cfg.decimation = a.physics_hz // a.control_hz
        env_cfg.sim.render_interval = 1  # render (and grab a frame) after every physics step
        # keep the configured latency in seconds: (0-2 steps at 60 Hz) -> steps at the control rate
        lo, hi = env_cfg.action_delay_steps_range
        env_cfg.action_delay_steps_range = (int(round(lo * a.control_hz / 60)), int(round(hi * a.control_hz / 60)))
        env_cfg.arm_increment_max = 0.05  # per step; the trajectory's rate * dt stays far below this
        env_cfg.episode_length_s = a.window + 2.0
        env_cfg.ball_spawn_x_range = (BALL_X, BALL_X)
        env_cfg.ball_spawn_y_range = (0.0, 0.0)
        env_cfg.arm_locked = False
        env_cfg.arm_monotonic = True
        env_cfg.base_forward_only = True
        env_cfg.viewer.resolution = (1920, 1080)
        if env_cfg.video_recorder is not None:  # rgb_array frames come from the video recorder's camera
            env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = 1920, 1080
        render = a.out != "none"
        env = gym.make(a.task, cfg=env_cfg, render_mode="rgb_array" if render else None).unwrapped
        REASONS = []
        orig_dones = type(env)._get_dones

        def dones_with_reasons(self):
            term, to = orig_dones(self)
            if term.any():
                oo = self.scene.env_origins
                ball_d = torch.linalg.norm((self.ball.data.root_pos_w.torch - oo)[:, :2], dim=-1)
                up = -self.robot.data.projected_gravity_b.torch[:, 2]
                bv = torch.linalg.norm(self.ball.data.root_lin_vel_w.torch, dim=-1)
                rv = torch.linalg.norm(self.robot.data.root_lin_vel_w.torch, dim=-1)
                for e in term.nonzero().flatten().tolist():
                    REASONS.append(dict(t=self.episode_length_buf[e].item() * self.step_dt, ball_d=ball_d[e].item(), up=up[e].item(),
                                        ball_v=bv[e].item(), robot_v=rv[e].item()))
            return term, to

        type(env)._get_dones = dones_with_reasons
        env.reset()
        dev, c = env.device, env.cfg
        o = env.scene.env_origins

        # random ball offset per robot (inside the alignment tolerance)
        dx = (torch.rand(n, device=dev) * 2 - 1) * a.ball_dx
        dy = (torch.rand(n, device=dev) * 2 - 1) * a.ball_dy
        ball_pose = env.ball.data.root_pose_w.torch.clone()
        ball_pose[:, 0] = o[:, 0] + BALL_X + dx
        ball_pose[:, 1] = o[:, 1] + c.ball_spawn_y_center + dy
        ball_pose[:, 2] = o[:, 2] + c.ball_radius
        env.ball.write_root_pose_to_sim_index(root_pose=ball_pose)
        env._ball_spawn_pos[:] = ball_pose[:, :3] - o

        # camera: oblique view over the whole grid
        center = o.mean(dim=0).tolist()
        span = a.grid * a.spacing
        lookat = (center[0] + 0.8, center[1], 0.0)
        eye = (center[0] - 0.55 * span, center[1] - 0.72 * span, 0.42 * span)
        env.sim.set_camera_view(eye=eye, target=lookat)
        for _ in range(5 if render else 0):
            env.sim.render()

        # optional state recording at every physics step (for replaying individual robots later)
        rec_root, rec_q, rec_ball = [], [], []
        if a.record:
            orig_update = env.scene.update

            def update_and_record(*args, **kwargs):
                orig_update(*args, **kwargs)
                rec_root.append((env.robot.data.root_pose_w.torch - torch.cat([o, torch.zeros_like(o[:, :1]).expand(-1, 4)], -1) * torch.tensor([1, 1, 1, 0, 0, 0, 0], device=dev)).half().clone())
                rec_q.append(env.robot.data.joint_pos.torch.half().clone())
                rec_ball.append((env.ball.data.root_pose_w.torch - torch.cat([o, torch.zeros_like(o[:, :1]).expand(-1, 4)], -1) * torch.tensor([1, 1, 1, 0, 0, 0, 0], device=dev)).float().clone())

            env.scene.update = update_and_record

        import imageio

        writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8) if render else None
        if render:
            # grab a frame right after every physics-step render inside env.step (control stays at control_hz)
            orig_render = env.sim.render

            def render_and_grab(*args, **kwargs):
                orig_render(*args, **kwargs)
                frame = env.video_recorder.render_rgb_array()
                if frame is not None and frame.mean() > 5:
                    writer.append_data(frame[..., :3])

            env.sim.render = render_and_grab
        default = env._default_arm_joint_pos
        dmax = torch.tensor([[T["d_shoulder"], T["d_elbow"]]], device=dev)
        press = T.get("press_elbow", 0.0)
        steps = int(round(a.window / env.step_dt))
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        streak = torch.zeros(n, device=dev)
        term_t = torch.full((n,), -1.0, device=dev)
        for k in range(steps):
            t = k * env.step_dt
            act = torch.zeros(n, 5, device=dev)
            act[:, 0] = (2.0 * T["v"] / c.base_lin_vel_scale - 1.0) if t < T["t_lift"] + T["t_brake"] else -1.0
            if t >= T["t_lift"]:
                remaining = (dmax - (env._arm_target - default)).clamp(min=0.0)
                act[:, 3:5] = (remaining.clamp(max=T["rate"] * env.step_dt) / c.arm_increment_max).clamp(0.0, 1.0)
            env._arm_ff_extra[:, 1] = press if t < T["t_lift"] else 0.0
            _, _, term, trunc, _ = env.step(act)
            # reasons are evaluated on the pre-reset state inside _get_dones (see the patch below)
            newly = alive & (term | trunc)
            term_t = torch.where(newly, torch.full_like(term_t, t), term_t)
            # record why: ball out of bounds vs other, before the reset moved things
            alive &= ~(term | trunc)
            innet = env._ball_in_net & alive
            streak = torch.where(innet, streak + 1, torch.zeros_like(streak))
            if k % 30 == 0:
                print(f"RENDER step {k}/{steps}", flush=True)
        if writer is not None:
            writer.close()
        ok = (streak >= int(round(1.0 / env.step_dt))) & alive
        clr = env.ball.data.root_pos_w.torch[:, 2] - o[:, 2] - c.ball_radius
        ball_xy = torch.linalg.norm((env.ball.data.root_pos_w.torch - o)[:, :2], dim=-1)
        stats = dict(dx=dx.tolist(), dy=dy.tolist(), ok=ok.tolist(), alive=alive.tolist(), final_clr=clr.tolist(),
                     ball_xy=ball_xy.tolist(), term_t=term_t.tolist())
        if a.record:
            import numpy as np

            np.savez_compressed(
                a.record, root=torch.stack(rec_root).cpu().float().numpy(), q=torch.stack(rec_q).cpu().float().numpy(),
                ball=torch.stack(rec_ball).cpu().numpy(), ok=ok.cpu().numpy(), alive=alive.cpu().numpy(),
                term_t=term_t.cpu().numpy(), final_clr=clr.cpu().numpy(), dx=dx.cpu().numpy(), dy=dy.cpu().numpy(),
                physics_dt=env.physics_dt, joint_names=np.array(env.robot.joint_names),
            )
            print(f"RENDER recorded {len(rec_root)} physics steps -> {a.record}")
        json.dump(stats, open((a.out if render else os.path.join(HERE, "array_stats")) + ".json", "w"))
        for r_ in REASONS[:12]:
            print(f"TERM t={r_['t']:.2f} ball_dist={r_['ball_d']:.2f} m up={r_['up']:.2f} ball_v={r_['ball_v']:.2f} robot_v={r_['robot_v']:.2f}")
        print(f"TERM total {len(REASONS)} | ball_dist>3: {sum(r_['ball_d'] > 3 for r_ in REASONS)} | tipped(up<0.5): {sum(r_['up'] < 0.5 for r_ in REASONS)} | ball_v>20: {sum(r_['ball_v'] > 20 for r_ in REASONS)} | robot_v>10: {sum(r_['robot_v'] > 10 for r_ in REASONS)}")
        print(f"RENDER done: {int(ok.sum())}/{n} robots holding the ball at the end; video {a.out}")
        env.close()


if __name__ == "__main__":
    main()
