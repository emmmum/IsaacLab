# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL on top of the hand-designed open-loop scoop trajectory, with a sparse success reward.

Every policy step the environment computes the nominal action of the open-loop trajectory (drive at ``v``, press the
lip down, lift both joints at ``rate`` from ``t_lift``, stop driving ``t_brake`` after the lift) and adds the policy's
scaled residual to it. A zero policy therefore reproduces the trajectory, so training starts from its ~97% success
rate and the sparse reward is informative from the first iteration.

Reward: ``rew_success`` once, at the time-out step, if the ball has been held in the net (stick contact, off the
floor, inside the pocket) for the last ``success_hold_s`` seconds; zero otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .robomaster_lacrosse_env import RobomasterLacrosseEnv
from .robomaster_lacrosse_scoop_rl_env_cfg import OBS_TERM_DIMS

if TYPE_CHECKING:
    from .robomaster_lacrosse_scoop_rl_env_cfg import RobomasterLacrosseScoopRLEnvCfg


class RobomasterLacrosseScoopRLEnv(RobomasterLacrosseEnv):
    cfg: RobomasterLacrosseScoopRLEnvCfg

    def __init__(self, cfg: RobomasterLacrosseScoopRLEnvCfg, render_mode: str | None = None, **kwargs):
        # obs_terms may have been overridden after the cfg's __post_init__, e.g. from the command line, where a
        # list arrives as the string "[a,b]"
        if isinstance(cfg.obs_terms, str):
            cfg.obs_terms = tuple(t.strip(" '\"") for t in cfg.obs_terms.strip("[]()").split(",") if t.strip())
        unknown = set(cfg.obs_terms) - set(OBS_TERM_DIMS)
        assert not unknown, f"unknown obs_terms {unknown}; choose from {list(OBS_TERM_DIMS)}"
        cfg.observation_space = sum(OBS_TERM_DIMS[t] for t in cfg.obs_terms)
        super().__init__(cfg, render_mode, **kwargs)
        # the base env's action buffers hold the full 5-D action (vx, vy, wz, shoulder, elbow), not the residual
        self._prev_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self._actions = torch.zeros(self.num_envs, 5, device=self.device)
        self._action_hist = torch.zeros(self.num_envs, self._max_delay + 1, 5, device=self.device)
        self._in_net_streak = torch.zeros(self.num_envs, device=self.device)
        self._hold_steps = int(round(self.cfg.success_hold_s / self.step_dt))
        self._d_max = torch.tensor([[self.cfg.traj_d_shoulder, self.cfg.traj_d_elbow]], device=self.device)
        self._success_ema = 0.0
        self._base_obs_scale = torch.ones(self.num_envs, device=self.device)
        self._base_vel_meas = torch.zeros(self.num_envs, device=self.device)
        self._base_dist_meas = torch.zeros(self.num_envs, device=self.device)

    def nominal_action(self) -> torch.Tensor:
        """Action of the open-loop trajectory at the current step, in the base env's 5-D action space."""
        c = self.cfg
        t = self.episode_length_buf.float() * self.step_dt
        act = torch.zeros(self.num_envs, 5, device=self.device)
        driving = t < c.traj_t_lift + c.traj_t_brake
        act[:, 0] = torch.where(driving, 2.0 * c.traj_v / c.base_lin_vel_scale - 1.0, -1.0)
        remaining = (self._d_max - (self._arm_target - self._default_arm_joint_pos)).clamp(min=0.0)
        lift = (remaining.clamp(max=c.traj_rate * self.step_dt) / c.arm_increment_max).clamp(0.0, 1.0)
        act[:, 3:5] = torch.where((t >= c.traj_t_lift)[:, None], lift, torch.zeros_like(lift))
        self._arm_ff_extra[:, 1] = torch.where(t < c.traj_t_lift, c.traj_press_elbow, 0.0)
        return act

    def expert_action(self) -> torch.Tensor:
        """Action of the open-loop trajectory in the policy's action space (the behavior-cloning label).

        Zero in residual mode. Otherwise (forward speed, shoulder, elbow), with ``bc_hold_label`` wherever the
        trajectory does not lift a joint. Call it before stepping: it reads the current step's time and arm target.
        """
        if self.cfg.residual:
            return torch.zeros(self.num_envs, 3, device=self.device)
        nominal = self.nominal_action()[:, [0, 3, 4]]
        arm = nominal[:, 1:]
        nominal[:, 1:] = torch.where(arm > 0.0, arm, torch.full_like(arm, self.cfg.bc_hold_label))
        return nominal

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        c = self.cfg
        if not c.residual:
            act = torch.zeros(self.num_envs, 5, device=self.device)
            act[:, [0, 3, 4]] = actions.clamp(-1.0, 1.0)
            self.nominal_action()  # keeps the lip press-down feed-forward of the trajectory (time-scheduled)
            super()._pre_physics_step(act)
            return
        residual = actions.clamp(-1.0, 1.0)
        act = self.nominal_action()
        # forward-only base: action a maps to speed 0.5 * (a + 1) * base_lin_vel_scale
        act[:, 0] += 2.0 * c.residual_speed_scale / c.base_lin_vel_scale * residual[:, 0]
        # the arm can only move up (monotonic targets), so zero-mean exploration noise would ratchet it upward and
        # lift the lip off the floor before the ball arrives; the lift therefore cannot start before t_lift
        lifting = (self.episode_length_buf.float() * self.step_dt >= c.traj_t_lift)[:, None]
        act[:, 3:5] += torch.where(lifting, c.residual_arm_scale * residual[:, 1:3], 0.0)
        super()._pre_physics_step(act)

    def _get_observations(self) -> dict:
        # inaccurate chassis odometry (only used by the base_vel / base_dist terms)
        vx = self.robot.data.root_lin_vel_b.torch[:, 0]
        noise = torch.randn_like(vx) * self.cfg.base_obs_vel_noise_std
        self._base_vel_meas = self._base_obs_scale * vx + noise
        self._base_dist_meas += self._base_vel_meas * self.step_dt
        terms = []
        for name in self.cfg.obs_terms:
            if name == "arm_pos":
                q = self.robot.data.joint_pos.torch[:, self._arm_joint_idx]
                terms.append(q - self._default_arm_joint_pos)
            elif name == "phase":
                terms.append((self.episode_length_buf.float() / self.max_episode_length)[:, None])
            elif name == "arm_vel":
                terms.append(self.robot.data.joint_vel.torch[:, self._arm_joint_idx])
            elif name == "arm_torque":
                terms.append(self.robot.data.applied_torque.torch[:, self._arm_joint_idx])
            elif name == "base_vel":
                terms.append(self._base_vel_meas[:, None])
            elif name == "base_dist":
                terms.append(self._base_dist_meas[:, None])
        obs = torch.cat(terms, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        super()._get_rewards()  # updates self._ball_in_net; its shaped reward is not used
        alive = ~self.reset_terminated
        self._in_net_streak = torch.where(
            self._ball_in_net & alive, self._in_net_streak + 1, torch.zeros_like(self._in_net_streak)
        )
        success = self.reset_time_outs & alive & (self._in_net_streak >= self._hold_steps)
        finished = self.reset_time_outs | self.reset_terminated
        if finished.any():
            rate = (success.sum() / finished.sum()).item()
            self._success_ema = 0.95 * self._success_ema + 0.05 * rate if self._success_ema > 0 else rate
            self.extras.setdefault("log", {})["success_rate"] = self._success_ema
        return self.cfg.rew_success * success.float()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._in_net_streak[env_ids] = 0.0
        self._base_obs_scale[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.base_obs_scale_range
        )
        self._base_vel_meas[env_ids] = 0.0
        self._base_dist_meas[env_ids] = 0.0
