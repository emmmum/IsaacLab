# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the residual-RL lacrosse scoop (see :mod:`.robomaster_lacrosse_scoop_rl_env`)."""

from isaaclab.utils.configclass import configclass

from .robomaster_lacrosse_env_cfg import RobomasterLacrosseEnvCfg

# observation terms and their sizes
OBS_TERM_DIMS = {
    "arm_pos": 2,  # shoulder, elbow angle relative to the start pose [rad]; monotonic, so it doubles as a phase
    "phase": 1,  # elapsed episode time / episode length
    "arm_vel": 2,  # shoulder, elbow velocity [rad/s]
    "arm_torque": 2,  # applied shoulder, elbow torque [N*m] (what the MIT motors report)
    "base_vel": 1,  # measured chassis forward speed [m/s] (inaccurate, see base_obs_*)
    "base_dist": 1,  # odometry: distance driven since the episode start [m], integrated from the measured speed
}


@configclass
class RobomasterLacrosseScoopRLEnvCfg(RobomasterLacrosseEnvCfg):
    """Residual-RL scoop: sparse success reward, nominal open-loop trajectory as the base action."""

    episode_length_s = 4.0  # scoop (3 s) + hold (1 s); ends by time-out unless the ball leaves the arena
    action_space = 3  # residuals: forward speed, shoulder lift increment, elbow lift increment
    obs_terms: tuple[str, ...] = ("arm_pos",)
    observation_space = 2  # set from obs_terms in __post_init__

    # nominal open-loop trajectory (logs/scoop_traj_search/best_trajectory_robust.json)
    traj_v = 0.9  # [m/s] drive speed from t = 0
    traj_t_lift = 0.9  # [s] lift start
    traj_rate = 1.0  # [rad/s] lift rate of each joint
    traj_d_shoulder = 0.3  # [rad] shoulder lift above the start pose
    traj_d_elbow = 0.7  # [rad] elbow lift above the start pose
    traj_t_brake = 1.0  # [s] stop driving this long after the lift starts
    traj_press_elbow = -1.0  # [N*m] elbow feed-forward pressing the lip down before the lift

    # True: action = nominal trajectory + residual. False: the policy outputs the actions itself (forward-speed
    # action in [-1, 1] -> 0..base_lin_vel_scale, shoulder / elbow lift increment as a fraction of arm_increment_max,
    # <= 0 holds the arm); the trajectory then only provides the behavior-cloning labels (see the env's expert_action)
    residual = True
    # BC label for "do not lift": any value <= 0 holds the arm, a margin keeps small regression errors from lifting it
    bc_hold_label = -0.5

    # residual scales: action 1.0 changes the forward speed by this much [m/s] / the lift increment by this
    # fraction of arm_increment_max per step (the nominal lift uses rate * dt / arm_increment_max = 0.33)
    residual_speed_scale = 0.2
    residual_arm_scale = 0.33

    # the chassis measurements are not accurate (unlike the arm encoders): per-episode scale error on the measured
    # speed plus white noise; the odometry distance integrates the measured speed, so its error grows with time
    base_obs_scale_range = (0.9, 1.1)
    base_obs_vel_noise_std = 0.05  # [m/s]

    rew_success = 1.0
    success_hold_s = 1.0  # [s] the ball must be in the net for this long at the end of the episode

    # robot at rest, ball ~1 m dead ahead with a small offset (the demo's distribution)
    ball_spawn_x_range = (0.97, 1.03)
    ball_spawn_y_range = (-0.01, 0.01)
    arm_locked = False
    arm_monotonic = True
    base_forward_only = True

    def __post_init__(self):
        super().__post_init__()
        unknown = set(self.obs_terms) - set(OBS_TERM_DIMS)
        assert not unknown, f"unknown obs_terms {unknown}; choose from {list(OBS_TERM_DIMS)}"
        self.observation_space = sum(OBS_TERM_DIMS[t] for t in self.obs_terms)
