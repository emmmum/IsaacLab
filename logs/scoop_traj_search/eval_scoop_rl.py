# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the residual scoop policy (or the zero residual = open-loop trajectory) on the sparse success metric.

Runs whole episodes from a synchronized reset and counts successes (ball held in the net for the last second).

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/eval_scoop_rl.py --checkpoint <model_*.pt> [env.<field>=<value> ...]
  ./isaaclab.sh -p logs/scoop_traj_search/eval_scoop_rl.py --zero [env.<field>=<value> ...]
"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Scoop-RL-Direct-v0")
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--episodes", type=int, default=2, help="episodes per env")
p.add_argument("--checkpoint", default="", help="RSL-RL model_*.pt")
p.add_argument("--zero", action="store_true", help="evaluate the zero residual (open-loop trajectory)")
p.add_argument("--label", default="")
p.add_argument("--stochastic", action="store_true", help="sample actions with the policy's noise (as in training)")
p.add_argument("--trace", default="", help="save the first episode's per-step mean signals to this .npz")
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

        env_cfg.scene.num_envs = a.num_envs
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=env_cfg))
        if a.zero:
            def policy(obs):
                return torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(a.checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            if a.stochastic:
                def policy(obs):
                    return runner.alg.actor(obs, stochastic_output=True)
        u = env.unwrapped
        u.reset()
        obs = env.get_observations()
        succ = done = 0
        # success per ball-distance bin (env-local spawn x of the episode that just ended)
        edges = torch.linspace(0.9, 1.1, 6, device=u.device)
        bin_succ = torch.zeros(5, device=u.device)
        bin_done = torch.zeros(5, device=u.device)
        rough = u.cfg.rough_floor
        lvl_succ = torch.zeros(u.cfg.rough_levels, device=u.device)
        lvl_done = torch.zeros(u.cfg.rough_levels, device=u.device)
        trace = {k: [] for k in ("q", "tau", "contact", "vx", "act")}
        with torch.inference_mode():
            for k in range(a.episodes * int(u.max_episode_length)):
                spawn_x = u._ball_spawn_pos[:, 0].clone()
                action = policy(obs)
                obs, rew, dones, _ = env.step(action)
                if a.trace and k < u.max_episode_length - 1:
                    trace["q"].append((u.robot.data.joint_pos.torch[:, u._arm_joint_idx] - u._default_arm_joint_pos).mean(0))
                    trace["tau"].append(u.robot.data.applied_torque.torch[:, u._arm_joint_idx].mean(0))
                    f = u.ball_contact.data.force_matrix_w_history.torch
                    trace["contact"].append((torch.linalg.norm(f, dim=-1).amax(dim=(1, 2, 3)) > 0.05).float().mean())
                    trace["vx"].append(u.robot.data.root_lin_vel_b.torch[:, 0].mean())
                    trace["act"].append(action.clamp(-1, 1).mean(0))
                succ += int((rew > 0).sum())
                done += int(dones.sum())
                b = (torch.bucketize(spawn_x, edges) - 1).clamp(0, 4)
                bin_done.index_add_(0, b, dones.float())
                bin_succ.index_add_(0, b, (rew > 0).float())
                if rough:
                    lvl_done.index_add_(0, u._floor_level, dones.float())
                    lvl_succ.index_add_(0, u._floor_level, (rew > 0).float())
        label = a.label or ("zero" if a.zero else a.checkpoint)
        print(f"EVAL {label}: {succ}/{done} = {100.0 * succ / max(done, 1):.1f}%")
        bins = " ".join(
            f"[{edges[i]:.2f},{edges[i + 1]:.2f}) {100.0 * bin_succ[i] / bin_done[i]:.0f}%"
            for i in range(5)
            if bin_done[i] > 0
        )
        print(f"BINS {label}: {bins}")
        if rough:
            c = u.cfg
            levels = " ".join(
                f"L{k}({1000 * c.rough_bump_rms_max * k / (c.rough_levels - 1):.2f}mm,"
                f"{c.rough_tilt_max_deg * k / (c.rough_levels - 1):.2f}deg) {100.0 * lvl_succ[k] / lvl_done[k]:.1f}%"
                for k in range(c.rough_levels)
            )
            print(f"LEVELS {label}: {levels}")
        if a.trace:
            import numpy as np

            np.savez(a.trace, dt=u.step_dt, **{k: torch.stack(v).cpu().numpy() for k, v in trace.items()})
        env.close()


if __name__ == "__main__":
    main()
