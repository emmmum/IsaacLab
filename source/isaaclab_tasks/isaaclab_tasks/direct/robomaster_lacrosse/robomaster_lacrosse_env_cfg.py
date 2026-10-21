# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.robomaster_lacrosse import ROBOMASTER_LACROSSE_WHEELED_CFG

# offset from the `lowerarm` body frame to the center of the lacrosse stick's pocket (net),
# for the long-stick asset. The elbow joint sits at (0, 0.016, 0.48) in the same frame, so
# the elbow-to-pocket distance is ~0.46 m. (The stick prim's own origin is at the butt cap,
# not the pocket.)
STICK_POCKET_OFFSET = (0.46, 0.016, 0.46)
# offset from the `lowerarm` body frame to the front lip (tip) of the head: 0.18 m beyond the pocket center along
# the elbow->pocket axis (elbow-to-tip is ~0.64 m). In the initial pose this point touches the ground at x ~= 0.51 m.
STICK_LIP_OFFSET = (0.640, 0.016, 0.452)
# elbow joint position in the `lowerarm` body frame
STICK_ELBOW_OFFSET = (0.0, 0.016, 0.48)


@configclass
class RobomasterLacrosseEnvCfg(DirectRLEnvCfg):
    # env
    # 360 Hz physics: the ball does not tunnel through the net up to 14 m/s relative speed
    # (120 Hz tunnels from 8 m/s, 240 Hz from 14 m/s); the policy still runs at 60 Hz.
    decimation = 6
    episode_length_s = 8.0
    # action: [vx, vy, wz, arm_joint_1, arm_joint_2]
    action_space = 5
    # observation: base lin/ang vel (6) + arm joint pos/vel (4) + pocket->ball vector (3)
    # + ball lin vel (3) + previous action (5)
    observation_space = 21
    state_space = 0

    # action scales
    base_lin_vel_scale = 1.0  # [m/s]
    base_ang_vel_scale = 1.5  # [rad/s]

    # base slip randomization (command -> executed velocity), resampled every episode
    # matched to the real base (user, 2026-10-16): speed tracks the command within +-10%, straight-line driving is
    # accurate (no lateral / yaw drift), acceleration ~1.7 m/s^2 (1.5 used, conservative)
    base_slip_scale_range = (0.9, 1.1)  # per-axis multiplier on the (vx, vy, wz) command
    base_slip_bias_max = (0.0, 0.0, 0.0)  # [m/s, m/s, rad/s] constant drift, uniform in +/- this
    base_slip_lag_tau_range = (0.05, 0.2)  # [s] first-order lag time constant
    base_slip_noise_std = (0.02, 0.0, 0.0)  # [m/s, m/s, rad/s] per-physics-step noise before the lag
    # acceleration limit on the executed base velocity (real robot: ~1.7 m/s^2, top speed ~2 m/s). None = no limit
    # (only the first-order lag above, which starts at ~4-16 m/s^2 from rest).
    base_max_lin_accel: float | None = 1.5  # [m/s^2] on the planar (vx, vy) velocity
    base_max_ang_accel: float | None = 4.0  # [rad/s^2] on the yaw rate (not measured; assumed)
    # arm joint-target range (the USD joints are unlimited). Action 0 is the default pose; negative actions scale
    # toward the lower bound and positive toward the upper bound. Order: (RevoluteJoint1 shoulder, RevoluteJoint2
    # elbow), negative = head down. Scoop-only range: shoulder -34..+23 deg, elbow -52..+23 deg. A wide symmetric
    # +/-1.5 rad range let the policy fold the arm to a corner (shoulder ~-98 deg) where it never lifted the ball.
    arm_joint_target_lower = (-0.593, -0.908)  # [rad]
    arm_joint_target_upper = (0.401, 0.401)  # [rad]
    # hold the arm at its default (scooping) pose and ignore the arm actions; used for curriculum stage 1, where
    # driving straight into the ball at ~0.9 m/s already rides it up onto the net
    arm_locked = False
    # monotonic arm: each policy step the arm actions (clamped to [0, 1]) raise the joint targets by up to
    # arm_increment_max; targets never move back down and stop at arm_joint_target_upper. After the ball is hit the
    # arm only ever needs to lift, and action <= 0 holds the arm still so exploration does not raise it early.
    arm_monotonic = True
    arm_increment_max = 0.05  # [rad] per policy step (3 rad/s at 60 Hz, below the 5 rad/s joint limit)
    # forward-only base: the vx action [-1, 1] maps to [0, base_lin_vel_scale] (action 0 = half speed forward)
    base_forward_only = True

    # real arm mass properties, written over the USD values at startup (the USD stick was 0.17 kg with its CoM 0.34 m
    # past the elbow). Assumes the shoulder motor is fixed to the chassis and the elbow motor sits at the elbow on
    # link 1.
    arm_real_inertia = True
    upperarm_mass = 0.679  # [kg] 3D-printed link 1 (0.358) + elbow motor (0.321)
    upperarm_com_from_shoulder = 0.221  # [m] along the 0.30 m link
    upperarm_inertia_perp = 0.0065  # [kg*m^2] about the CoM, perpendicular to the link
    upperarm_inertia_axial = 0.0005  # [kg*m^2] about the link axis
    stick_mass = 0.457  # [kg]
    stick_length = 1.146  # [m] from the USD mesh (0.504 m behind the elbow to the tip 0.642 m ahead)
    stick_com_from_elbow = 0.142  # [m] CoM is 0.5 m from the tip
    stick_inertia_axial = 0.0005  # [kg*m^2]
    # MIT-mode feedforward: gravity compensation torque for the arm joints (from the simulator's model). Extra
    # per-env feedforward (e.g. a lip press-down torque) can be written to `env._arm_ff_extra`.
    # base mass properties: everything except the upper arm and stick weighs `base_total_mass` (estimate from the
    # DJI RoboMaster EP Core spec, ~3.3 kg with the stock arm). The shoulder motor sits on the shoulder `Cylinder`
    # body, the chassis `base` body takes the remainder (USD had 9.47 kg); its inertia is scaled by the mass ratio.
    base_real_mass = True
    base_total_mass = 3.0  # [kg] chassis + battery + wheels + mount + shoulder motor
    shoulder_motor_mass = 0.321  # [kg]
    arm_gravity_comp = True

    # ---- uncertainty (all off by default; ranges are sampled uniformly) ----
    # arm joint friction as explicit torque: tau_f = -coulomb * tanh(qd / 0.02) - viscous * qd, sampled per joint
    # per episode. It enters through the feedforward path, so it is clipped together with the motor torque.
    arm_friction_coulomb_range = (0.0, 0.0)  # [N*m]
    arm_friction_viscous_range = (0.0, 0.0)  # [N*m*s/rad]
    # gravity-compensation model error: per-joint scale on the feedforward, sampled per episode
    arm_gravity_comp_scale_range = (1.0, 1.0)
    # control latency: actions take effect this many policy steps late (integer, sampled per episode)
    action_delay_steps_range = (0, 0)
    # contact materials, sampled once per env at startup (None = keep the nominal values)
    ball_static_friction_range: tuple[float, float] | None = None
    ball_dynamic_friction_ratio = 0.9  # dynamic = ratio * static
    ball_restitution_range: tuple[float, float] | None = None
    robot_friction_scale_range: tuple[float, float] | None = (
        None  # scales static + dynamic friction of all robot shapes
    )

    # ---- floor and ball imperfections (off by default) ----
    # Floor that is flat but not level: when floor_tilt_max_deg > 0 a kinematic platform replaces the global ground
    # under each env (the ground plane is lowered out of the way) and gets a random roll/pitch in +- this range every
    # episode, pivoting at the robot's start point. Friction matches the default ground plane.
    floor_tilt_max_deg = 0.0  # [deg]
    floor_size = (3.5, 2.0, 0.1)  # [m] length (x), width (y), thickness
    floor_center_x = 1.0  # [m] platform center ahead of the robot start point
    floor_friction = 0.5
    # Uneven floor (rough_floor = True; replaces the ground plane, excludes floor_tilt_max_deg): a triangle-mesh terrain
    # of rough_levels difficulty rows x one tile per env (see rough_floor.py; envs must not share a tile: overlapping
    # sticks make PhysX collide their SDFs pairwise, which is extremely slow); level k has k / (levels - 1) of the
    # maximum bump height and tilt, and env e uses level e % rough_levels. The defaults aim at an indoor lab floor
    # (sub-millimeter bumps, a few tenths of a degree of slope). At startup each env's ball is dropped at
    # random points within rough_ball_region of its tile center and simulated until it comes to rest (several rounds);
    # every episode then puts the ball at one of those rest positions and the robot behind it, so that the ball sits
    # ball_spawn_x/y ahead of it (centered on the stick head), with the robot's height and roll/pitch fitted to the
    # floor under its wheels.
    rough_floor = False
    rough_levels = 8
    rough_tile_size = (3.2, 1.0)  # [m] length along the driving direction (x), width (y)
    rough_spacing = 0.04  # [m] height-map / mesh resolution
    rough_bump_rms_max = 0.001  # [m] RMS bump height at the top level
    rough_bump_corr = 0.15  # [m] bump correlation length
    rough_tilt_max_deg = 0.3  # [deg] tile tilt at the top level
    rough_seed = 0
    rough_ball_region = (0.3, 0.3)  # [m] half-size of the ball drop region around the tile center
    rough_settle_rounds = 6
    rough_settle_s = 1.5  # [s] simulated per drop round
    rough_rest_speed = 0.003  # [m/s] a ball slower than this at the end of a round counts as at rest
    # Rolling resistance of the ball on the floor (PhysX has none, so a ball on any slope would roll forever): while the
    # ball touches the floor, a torque of ball_rolling_resistance * m * g * r opposes its rolling, smoothed by
    # tanh(|omega| / ball_rolling_omega0). A ball then stays put on slopes below ~asin(resistance).
    ball_rolling_resistance = 0.015
    ball_rolling_omega0 = 0.05  # [rad/s]
    # Imperfect ball: mass and CoM offset are sampled once per env at startup (the ball is placed resting CoM-down);
    # vary the size with ball_radius + ball_cfg.spawn.radius. EXPERIMENTAL: "ellipsoid" spawns an icosphere mesh scaled
    # by ball_ellipsoid_scale, but PhysX convex-mesh contact lets it rest 2-5 mm into the floor and even a round mesh
    # ball scores ~85% vs 100% for the exact sphere, so results with it reflect that artifact, not out-of-roundness.
    ball_shape = "sphere"  # "sphere" | "ellipsoid"
    ball_ellipsoid_scale = (1.03, 1.0, 0.98)
    ball_mass_range: tuple[float, float] | None = None  # [kg]
    ball_com_offset_max = 0.0  # [m] CoM offset in a random direction

    # simulation
    # the stick head rests on the ground with SDF collision, which needs far more GPU collision stack than the
    # 2**26 default at 8192 envs (PhysX asked for ~3.6e8 bytes and dropped contacts otherwise)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 360, render_interval=decimation, physics=PhysxCfg(gpu_collision_stack_size=2**29)
    )

    # robot
    # initial scooping pose: the stick points 48 deg below horizontal and the head's front lip just
    # touches the ground (upper arm tilted ~14 deg forward). Joint angles are negative head-down.
    # wheel-driven base: the chassis is the articulation root (its frame sits at ground level when the wheels
    # touch the ground) and the four cylinder wheels are velocity-controlled, see `base_drive`
    robot_cfg: ArticulationCfg = ROBOMASTER_LACROSSE_WHEELED_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ROBOMASTER_LACROSSE_WHEELED_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.005),
            joint_pos={"RevoluteJoint1": -0.246, "RevoluteJoint2": -0.592, "RevoluteJoint(_[0-2])?$": 0.0},
        ),
    )
    # "wheels": the (vx, wz) command (after slip/accel model) becomes wheel velocity targets; vy is ignored because
    # cylinder wheels cannot strafe. "root_velocity": legacy mode that overwrites the root velocity every physics step.
    base_drive = "wheels"
    wheel_radius = 0.0512  # [m]
    # exact cylinder wheel colliders (PhysX custom geometry, contacts computed on the CPU); False uses the convex
    # approximation
    exact_cylinder_wheels = True
    wheel_joint_names = ["RevoluteJoint", "RevoluteJoint_0", "RevoluteJoint_1", "RevoluteJoint_2"]
    arm_joint_names = ["RevoluteJoint1", "RevoluteJoint2"]

    # ball (procedural sphere -- no external USD asset available for a lacrosse ball)
    ball_radius = 0.0325  # [m], standard lacrosse ball
    ball_mass = 0.141  # [kg], standard lacrosse ball
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, ball_radius)),
    )

    # contact between the ball and the stick (all stick/net colliders live on the `lowerarm` body)
    ball_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Ball",
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/lowerarm"],
        history_length=decimation,
    )
    net_contact_force_threshold = 0.05  # [N] ball weight is ~1.4 N
    # the ground is a single global plane, so ground contact is detected geometrically: flat ground means the
    # ball touches it exactly when its bottom is at floor height
    ground_contact_tolerance = 0.003  # [m]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    # ball spawn randomization (relative to robot base, on the ground): about 1 m ahead, both robot and ball at
    # rest, so the robot has to build up scooping speed itself (the head's front lip is at x ~= 0.51 m)
    ball_spawn_x_range = (0.8, 1.2)
    # lateral spawn range around ball_spawn_y_center; the stick head is not on the robot's center line (its lip is
    # STICK_LIP_OFFSET[1] = 1.6 cm to the left), so balls are centered on the stick head by default
    ball_spawn_y_range = (-0.15, 0.15)
    ball_spawn_y_center = STICK_LIP_OFFSET[1]  # [m]

    # initial base velocity (body frame), sampled every episode
    base_init_vel_x_range = (0.0, 0.0)  # [m/s]
    base_init_vel_y_range = (0.0, 0.0)  # [m/s]
    base_init_yaw_rate_range = (0.0, 0.0)  # [rad/s]

    # reward curriculum (select with the hydra override `env.reward_stage=1`):
    #   1 -- pop the ball: lift/raise rewards whenever the ball is off the ground near the pocket (no contact needed)
    #   2 -- pocket it: lift/raise rewards only while the ball also touches the stick (fine-tune from a stage-1 run)
    # Stage 1 is meant to be run with the arm locked: `env.reward_stage=1 env.arm_locked=True`.
    reward_stage = 2

    # reward scales
    rew_scale_reach = 1.0
    rew_scale_lift = 5.0
    rew_scale_raise = 2.0
    # one-off penalty when an episode ends early (tipped over, ball out of bounds, blow-up); without it the policy
    # learned to tip itself over to escape the action-rate penalty
    rew_scale_terminated = -20.0
    rew_scale_action_rate = -0.01

    # reach reward shaping
    # reach is measured from the head's front lip to the ball's ground-contact point, so it pays only for the
    # scoop-ready posture (lip down at the ball). Measuring from the pocket center let the policy cover the ball
    # from above with the arm folded at its limits, which can never lift it. Pinned to 1 while the ball is in the
    # rewarded state so lifting never costs reach reward.
    reach_std = 0.1
    raise_std = 0.04  # [m] height shaping saturates at ~0.1 m clearance: higher is not better beyond that
    # lift quality: the lift/raise rewards are scaled by exp(-forward / pop_forward_scale) * exp(-(v / scale)^2),
    # where forward is how far the ball has moved ahead of its spawn point (along the robot's initial heading) and
    # v is the robot base's planar speed -- a clean pop keeps the ball in place and the robot slow
    pop_forward_scale = 0.3  # [m]
    pop_robot_speed_scale = 1.0  # [m/s]
    lift_height_threshold = 0.08  # [m] above ground for "scooped" bonus
    lift_containment_radius = 0.15  # [m] horizontal distance from pocket to count as "contained"
    # lift/raise rewards are only paid while the ball is "in the net": touching the stick, off the ground,
    # and within lift_containment_radius of the pocket (so resting on the shaft does not count)

    # termination bounds
    arena_bound = 3.0  # [m] ball/robot beyond this from env origin ends the episode
    tilt_threshold = 0.5  # min allowed projected-gravity z-component (robot "up") before ending episode
    max_ball_speed = 20.0  # [m/s] faster than this is treated as a physics blow-up and ends the episode
    max_robot_speed = 10.0  # [m/s] same, for the robot base
