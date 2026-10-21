# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior-clone the open-loop scoop trajectory into the PPO actor MLP, then save it as an RSL-RL checkpoint.

1. Roll out the trajectory (the env's ``expert_action``) in the pure-MLP mode (``env.residual=False``) and record
   (observation, expert action) pairs.
2. Fit the actor's observation normalizer and MLP to them (MSE on the deterministic output).
3. Evaluate the cloned policy on the sparse success metric, and save ``model_0.pt`` so PPO can fine-tune it:
   ``train.py --task <task> --resume --load_run <run dir name> --checkpoint model_0.pt env.residual=False ...``

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/bc_scoop.py --run_name bc_phase 'env.obs_terms=[arm_pos,phase]'
"""

import argparse
import datetime
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Scoop-RL-Direct-v0")
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--collect_episodes", type=int, default=2, help="expert episodes per env")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch", type=int, default=8192)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--init_std", type=float, default=0.1, help="action noise std stored for PPO fine-tuning")
p.add_argument("--run_name", default="bc")
add_launcher_args(p)
a, rem = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + rem


def rollout(env, act_fn, steps):
    """Step the env with ``act_fn(obs)``; return (#successes, #finished episodes) and the visited (obs, label)."""
    u = env.unwrapped
    obs = env.get_observations()
    succ = done = 0
    obs_buf, lab_buf = [], []
    for _ in range(steps):
        label = u.expert_action()
        obs_buf.append(obs["policy"].clone())
        lab_buf.append(label.clone())
        obs, rew, dones, _ = env.step(act_fn(obs, label))
        succ += int((rew > 0).sum())
        done += int(dones.sum())
    return succ, done, torch.cat(obs_buf), torch.cat(lab_buf)


@hydra_task_config(a.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, a):
        import importlib.metadata as metadata

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = a.num_envs
        env_cfg.residual = False
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=env_cfg))
        u = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        actor, critic = runner.alg.actor, runner.alg.critic
        ep_steps = int(u.max_episode_length)

        # 1. expert data
        u.reset()
        with torch.no_grad():
            succ, done, X, Y = rollout(env, lambda obs, label: label, a.collect_episodes * ep_steps)
        print(f"BC expert rollout: {succ}/{done} = {100.0 * succ / max(done, 1):.1f}% success, {len(X)} samples")
        X, Y = X.clone(), Y.clone()

        # 2. normalizers from the data, then MSE on the deterministic actor output
        for model in (actor, critic):
            for chunk in X.split(65536):
                model.update_normalization({"policy": chunk})
            model.obs_normalizer.eval()  # freeze the statistics while fitting
        opt = torch.optim.Adam(actor.mlp.parameters(), lr=a.lr)
        n = len(X)
        for epoch in range(a.epochs):
            perm = torch.randperm(n, device=X.device)
            tot = 0.0
            for i in range(0, n, a.batch):
                idx = perm[i : i + a.batch]
                pred = actor({"policy": X[idx]})
                loss = torch.nn.functional.mse_loss(pred, Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if epoch % 5 == 0 or epoch == a.epochs - 1:
                print(f"BC epoch {epoch}: mse {tot / n:.5f}")
        for model in (actor, critic):
            model.obs_normalizer.train()
        with torch.no_grad():
            actor.distribution.std_param.fill_(a.init_std)

        # 3. evaluate the cloned policy (deterministic) and save it as a PPO checkpoint
        u.reset()
        with torch.no_grad():
            succ, done, _, _ = rollout(env, lambda obs, label: actor(obs), 2 * ep_steps)
        print(f"BC policy rollout: {succ}/{done} = {100.0 * succ / max(done, 1):.1f}% success")
        log_dir = os.path.join(
            "logs", "rsl_rl", agent_cfg.experiment_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + a.run_name
        )
        os.makedirs(log_dir, exist_ok=True)
        # same content as runner.save(); that also calls the logger, which has no writer without a log dir
        torch.save({**runner.alg.save(), "iter": 0, "infos": None}, os.path.join(log_dir, "model_0.pt"))
        print(f"BC saved {log_dir}/model_0.pt")
        env.close()


if __name__ == "__main__":
    main()
