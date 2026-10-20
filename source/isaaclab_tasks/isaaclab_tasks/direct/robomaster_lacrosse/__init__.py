# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RoboMaster mobile-manipulator lacrosse-ball-scoop environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-RoboMaster-Lacrosse-Direct-v0",
    entry_point=f"{__name__}.robomaster_lacrosse_env:RobomasterLacrosseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robomaster_lacrosse_env_cfg:RobomasterLacrosseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobomasterLacrossePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoboMaster-Lacrosse-Scoop-RL-Direct-v0",
    entry_point=f"{__name__}.robomaster_lacrosse_scoop_rl_env:RobomasterLacrosseScoopRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robomaster_lacrosse_scoop_rl_env_cfg:RobomasterLacrosseScoopRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobomasterLacrosseScoopRLPPORunnerCfg",
    },
)
