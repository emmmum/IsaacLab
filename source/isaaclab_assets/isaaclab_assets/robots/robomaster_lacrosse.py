# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RoboMaster-Lacrosse mobile manipulator.

The following configurations are available:

* :obj:`ROBOMASTER_LACROSSE_CFG`: DJI RoboMaster-style wheeled base with a 2-DOF
  arm holding a lacrosse stick as the end-effector.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

ROBOMASTER_LACROSSE_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "robomaster_lacrosse", "robomaster_lacrosse.usd"
)

ROBOMASTER_LACROSSE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOMASTER_LACROSSE_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # the base has no dedicated drive joints -- it is a genuine free 6-dof root body,
        # commanded directly via write_root_velocity_to_sim in the task's _apply_action
        # (rotating a body-frame (vx, vy, wz) command into world frame by current yaw).
        # the wheel joints get light passive damping so they roll along cosmetically
        # rather than spinning freely / adding noise.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["RevoluteJoint(_[0-2])?$"],
            stiffness=0.0,
            damping=10.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["RevoluteJoint[12]"],
            stiffness=200.0,
            damping=20.0,
            effort_limit_sim=50.0,
        ),
    },
)
"""Configuration for the RoboMaster-Lacrosse mobile manipulator.

The following control configuration is used:

* Base: holonomic (vx, vy, wz) velocity command written directly to the free root body
  (see the task's action term) -- the wheel joints are passive/cosmetic only.
* Arm: position control with damping.
"""

ROBOMASTER_LACROSSE_LONG_STICK_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "robomaster_lacrosse", "robomaster_lacrosse_long_stick.usd"
)

ROBOMASTER_LACROSSE_LONG_STICK_CFG = ROBOMASTER_LACROSSE_CFG.replace(
    spawn=ROBOMASTER_LACROSSE_CFG.spawn.replace(usd_path=ROBOMASTER_LACROSSE_LONG_STICK_USD_PATH),
    actuators={
        **ROBOMASTER_LACROSSE_CFG.actuators,
        # 5 rad/s per joint keeps the pocket at ~10 m/s or below: with 6 rad/s, random full-scale
        # actions (base + arm combined) still reached ~13 m/s at the 95th percentile.
        # real arm motors run in MIT mode: tau = Kp (q_des - q) + Kd (qd_des - qd) + tau_ff, clipped to 4 N*m, so the
        # explicit IdealPD model (clips the total including feedforward) is used instead of the PhysX drive.
        # Kd = 3 N*m*s/rad; Kp sized for a damping ratio of ~1.2 at the real link inertias (shoulder 0.118-0.155,
        # elbow 0.059 kg*m^2): shoulder 10 N*m/rad, elbow 25 N*m/rad.
        "arm": IdealPDActuatorCfg(
            joint_names_expr=["RevoluteJoint[12]"],
            stiffness={"RevoluteJoint1": 10.0, "RevoluteJoint2": 25.0},
            damping=3.0,
            effort_limit=4.0,
            velocity_limit=5.0,
        ),
    },
)
"""Same as :obj:`ROBOMASTER_LACROSSE_CFG`, with the lacrosse stick slid 0.08 m forward along its
shaft so that the head sits farther from the elbow, matching the physical robot.

Elbow-to-pocket distance is 0.46 m (0.38 m in the original) and elbow-to-tip distance is 0.64 m.
The stick mass is set to 0.35 kg and the arm joints are limited to 5 rad/s.
"""

ROBOMASTER_LACROSSE_WHEELED_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "robomaster_lacrosse", "robomaster_lacrosse_long_stick_wheeled.usd"
)

ROBOMASTER_LACROSSE_WHEELED_CFG = ROBOMASTER_LACROSSE_LONG_STICK_CFG.replace(
    spawn=ROBOMASTER_LACROSSE_LONG_STICK_CFG.spawn.replace(usd_path=ROBOMASTER_LACROSSE_WHEELED_USD_PATH),
    # the articulation root is the chassis, whose frame sits at ground level when the wheels touch the ground
    init_state=ROBOMASTER_LACROSSE_LONG_STICK_CFG.init_state.replace(pos=(0.0, 0.0, 0.005)),
    actuators={
        **ROBOMASTER_LACROSSE_LONG_STICK_CFG.actuators,
        # wheel motors in velocity mode: damping is the velocity gain [N*m*s/rad]. Torque limit back-computed from the
        # real ~1.5 m/s^2 acceleration: 4.14 kg * 1.5 m/s^2 / 4 wheels * 0.0512 m = 0.08 N*m, plus lip drag and margin
        # -> 0.15 N*m (~2.5 m/s^2 max on flat ground). 40 rad/s = ~2 m/s top speed.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["RevoluteJoint(_[0-2])?$"],
            stiffness=0.0,
            damping=2.0,
            effort_limit_sim=0.15,
            velocity_limit_sim=40.0,
        ),
    },
)
"""Same as :obj:`ROBOMASTER_LACROSSE_LONG_STICK_CFG`, driven by its wheels instead of a root-velocity override.

The chassis is the articulation root, each wheel collides as a plain cylinder (r = 0.0512 m, width 0.045 m) and the
four wheel joints are velocity-controlled. Cylinders cannot strafe: the base moves forward/backward and turns by skid
steering. Built by ``data/robomaster_lacrosse/make_wheeled_usd.py``.
"""
