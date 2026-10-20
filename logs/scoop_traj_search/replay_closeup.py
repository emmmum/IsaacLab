# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded robots (from render_array.py --record) as close-up videos.

The recorded per-physics-step chassis pose, joint angles and ball pose are written back into a fresh simulation
every frame (physics is not bit-for-bit repeatable, so re-simulating a failure might not fail again), and a camera
follows each robot from the side. Each clip is written at --fps from one frame per physics step (1200 Hz -> 240 fps
is 5x slow motion), with a label overlay.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/replay_closeup.py --headless --enable_cameras \
      --record logs/scoop_traj_search/failure_record.npz --envs 145 172 --labels "A" "B" --out_dir <dir>
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Direct-v0")
p.add_argument("--record", required=True)
p.add_argument("--envs", type=int, nargs="+", required=True)
p.add_argument("--labels", nargs="+", default=None)
p.add_argument("--t_start", type=float, default=0.55)
p.add_argument("--t_end", type=float, default=2.1)
p.add_argument("--fps", type=int, default=240)
p.add_argument("--out_dir", required=True)
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # load every array once: indexing an NpzFile re-reads and decompresses the whole array on each access
    with np.load(a.record) as f:
        rec = {k: f[k] for k in f.files}
    dt = float(rec["physics_dt"])
    with launch_simulation(env_cfg, a):
        n = len(a.envs)
        env_cfg.scene.num_envs = n
        env_cfg.sim.dt = dt
        env_cfg.decimation = 1
        env_cfg.sim.render_interval = 1
        env_cfg.episode_length_s = 60.0
        env_cfg.viewer.resolution = (1280, 720)
        if env_cfg.video_recorder is not None:
            env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = 1280, 720
        env = gym.make(a.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
        env.reset()
        dev = env.device
        o = env.scene.env_origins
        assert list(rec["joint_names"]) == list(env.robot.joint_names), "joint order changed since recording"

        import imageio
        from PIL import Image, ImageDraw, ImageFont

        font_big = ImageFont.truetype(FONT, 31)
        font_small = ImageFont.truetype(FONT, 21)
        os.makedirs(a.out_dir, exist_ok=True)
        ids = torch.arange(n, device=dev)
        zeros6 = torch.zeros(n, 6, device=dev)
        k0, k1 = int(a.t_start / dt), int(a.t_end / dt)
        for j, e in enumerate(a.envs):
            label = a.labels[j] if a.labels else f"env {e}"
            out = os.path.join(a.out_dir, f"closeup_env{e}.mp4")
            writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
            term_k = int(rec["term_t"][e] / dt) - 2 if rec["term_t"][e] > 0 else rec["root"].shape[0]
            # fixed side camera for this clip, centred on the robot's path over the clip (moving the camera every
            # frame made rendering ~10x slower)
            kk = np.arange(k0, min(k1, term_k))
            xs = rec["root"][kk, e, 0]
            x_mid, y_mid = 0.5 * (xs.min() + xs.max()), float(rec["root"][kk, e, 1].mean())
            cx, cy = x_mid + o[j, 0].item(), y_mid + o[j, 1].item()
            # the video recorder captures from its own camera prim and only applies cfg eye/lookat on its first
            # capture, so aim that camera directly (and update its cfg so the first capture agrees)
            from isaacsim.core.rendering_manager import ViewportManager

            cap = env.video_recorder._capture
            eye, target = (cx + 0.45, cy - 2.6, 0.9), (cx + 0.45, cy, 0.38)
            cap.cfg.eye, cap.cfg.lookat = eye, target
            env.video_recorder.render_rgb_array()
            ViewportManager.set_camera_view(cap.cfg.camera_prim_path, eye=list(eye), target=list(target))
            for _ in range(3):
                env.sim.render()
                env.video_recorder.render_rgb_array()
            for k in range(k0, min(k1, term_k)):
                # write every replayed robot (all envs share the recorded frame index)
                root = torch.tensor(rec["root"][k, a.envs], device=dev)
                q = torch.tensor(rec["q"][k, a.envs], device=dev)
                ball = torch.tensor(rec["ball"][k, a.envs], device=dev)
                root[:, :3] += o
                ball[:, :3] += o
                env.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
                env.robot.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)
                env.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
                env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
                env.ball.write_root_pose_to_sim_index(root_pose=ball, env_ids=ids)
                env.ball.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)
                env.sim.step(render=False)
                env.scene.update(dt=dt)
                env.sim.render()
                frame = env.video_recorder.render_rgb_array()
                if frame is None or frame.mean() < 5:
                    continue
                img = Image.fromarray(np.ascontiguousarray(frame[..., :3]))
                draw = ImageDraw.Draw(img)
                t = k * dt
                draw.rectangle([0, 0, img.width, 94], fill=(18, 19, 21))
                draw.text((26, 12), label, font=font_big, fill=(242, 194, 48))
                draw.text((26, 56), f"t = {t:4.2f} s   ·   1200 Hz 物理，{round(1 / (dt * a.fps))}× 慢放   ·   机器人 #{e}", font=font_small, fill=(236, 236, 234))
                writer.append_data(np.asarray(img))
            writer.close()
            print(f"REPLAY wrote {out}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
