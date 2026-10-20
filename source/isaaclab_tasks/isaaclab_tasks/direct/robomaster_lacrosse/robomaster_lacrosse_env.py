# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    sample_uniform,
)

from .robomaster_lacrosse_env_cfg import STICK_ELBOW_OFFSET, STICK_LIP_OFFSET, STICK_POCKET_OFFSET

if TYPE_CHECKING:
    from .robomaster_lacrosse_env_cfg import RobomasterLacrosseEnvCfg


class RobomasterLacrosseEnv(DirectRLEnv):
    cfg: RobomasterLacrosseEnvCfg

    def __init__(self, cfg: RobomasterLacrosseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._shoulder_body_idx = self.robot.find_bodies("Cylinder")[0][0]  # the shoulder joint sits at its origin
        if self.cfg.base_drive == "wheels":
            self._wheel_joint_idx, _ = self.robot.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
            # lateral position of each wheel in the chassis frame (left = +y), from the wheel links' child bodies
            wheel_body_ids = [
                self.robot.find_bodies(f"{side}_wheel_link")[0][0]
                for side in ("rear_left", "rear_right", "front_left", "front_right")
            ]
            rel_w = self.robot.data.body_pos_w.torch[0, wheel_body_ids] - self.robot.data.root_pos_w.torch[0]
            rel_b = quat_apply_inverse(self.robot.data.root_quat_w.torch[0].expand(4, -1), rel_w)
            self._wheel_y = rel_b[:, 1].clone()
        self._lowerarm_body_idx, _ = self.robot.find_bodies("lowerarm")
        self._lowerarm_body_idx = self._lowerarm_body_idx[0]

        self._pocket_offset = torch.tensor([STICK_POCKET_OFFSET], device=self.device)
        self._lip_offset = torch.tensor([STICK_LIP_OFFSET], device=self.device)

        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        self._default_arm_joint_pos = self.robot.data.default_joint_pos.torch[:, self._arm_joint_idx].clone()

        # base slip randomization: per-episode (vx, vy, wz) scale and bias, a first-order lag between
        # the command and the executed velocity, and per-step noise. See _apply_action.
        self._base_cmd_scale = torch.ones(self.num_envs, 3, device=self.device)
        self._base_cmd_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_lag_alpha = torch.ones(self.num_envs, 1, device=self.device)
        self._base_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._ball_in_net = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._arm_target = self._default_arm_joint_pos.clone()
        self._ball_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)  # env-local
        # MIT-mode arm commands: velocity target follows the monotonic target ramp; feedforward = gravity comp + extra
        self._arm_vel_target = torch.zeros_like(self._arm_target)
        self._arm_ff_extra = torch.zeros_like(self._arm_target)
        # uncertainty buffers (see the cfg); nominal values unless the ranges are widened
        self._arm_fric_coulomb = torch.zeros_like(self._arm_target)
        self._arm_fric_viscous = torch.zeros_like(self._arm_target)
        self._gc_scale = torch.ones_like(self._arm_target)
        self._max_delay = int(self.cfg.action_delay_steps_range[1])
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_hist = torch.zeros(self.num_envs, self._max_delay + 1, self.cfg.action_space, device=self.device)
        if self.cfg.arm_real_inertia:
            self._set_real_arm_inertia()
        if self.cfg.base_real_mass:
            self._set_real_base_mass()
        self._ball_com_offset = torch.zeros(self.num_envs, 3, device=self.device)  # body frame
        self._floor_normal = torch.zeros(self.num_envs, 3, device=self.device)
        self._floor_normal[:, 2] = 1.0
        self._randomize_materials()
        self._randomize_ball_inertia()

    def _set_real_base_mass(self):
        """Set the shoulder motor mass and scale the chassis so the base totals `base_total_mass`."""
        cfg, r = self.cfg, self.robot
        arm_ids = [r.find_bodies("upperarm")[0][0], self._lowerarm_body_idx]
        sh_idx = self._shoulder_body_idx
        chassis_idx = r.find_bodies("base")[0][0]
        masses = r.data.body_mass.torch.clone()
        inertias = r.data.body_inertia.torch.clone()
        other = [i for i in range(r.num_bodies) if i not in arm_ids + [sh_idx, chassis_idx]]
        chassis_mass = cfg.base_total_mass - cfg.shoulder_motor_mass - masses[0, other].sum().item()
        assert chassis_mass > 0.0, "base_total_mass is smaller than the wheels + mount"
        for idx, new_mass in ((sh_idx, cfg.shoulder_motor_mass), (chassis_idx, chassis_mass)):
            inertias[:, idx] *= new_mass / masses[:, idx : idx + 1]
            masses[:, idx] = new_mass
        ids = [sh_idx, chassis_idx]
        r.set_masses_index(masses=masses[:, ids], body_ids=ids)
        r.set_inertias_index(inertias=inertias[:, ids], body_ids=ids)

    def _randomize_ball_inertia(self):
        """Per-env ball mass and CoM offset (imperfect ball), sampled once at startup."""
        n = self.num_envs
        if self.cfg.ball_mass_range is not None:
            masses = sample_uniform(*self.cfg.ball_mass_range, (n, 1), self.device)
            self.ball.set_masses_index(masses=masses, body_ids=[0])
        if self.cfg.ball_com_offset_max > 0.0:
            d = torch.randn(n, 3, device=self.device)
            d = (
                d
                / torch.linalg.norm(d, dim=-1, keepdim=True)
                * sample_uniform(0.0, self.cfg.ball_com_offset_max, (n, 1), self.device)
            )
            self._ball_com_offset = d.clone()
            # RigidObject.set_coms_index currently fails on a float32/transformf buffer mismatch, so write the CoM
            # through the PhysX view directly (model properties live on the CPU)
            import warp as wp

            coms = wp.to_torch(self.ball.root_view.get_coms()).clone().reshape(n, -1)
            coms[:, :3] = d.cpu()
            env_ids = torch.arange(n, dtype=torch.int32)
            self.ball.root_view.set_coms(wp.from_torch(coms.contiguous(), dtype=wp.float32), wp.from_torch(env_ids))

    def _randomize_materials(self):
        """Sample contact materials once per env (PhysX material properties live on the CPU)."""
        import warp as wp

        def sample(rng, n):
            return torch.empty(n).uniform_(rng[0], rng[1])

        n = self.num_envs
        env_ids = torch.arange(n, dtype=torch.int32)
        if self.cfg.ball_static_friction_range is not None or self.cfg.ball_restitution_range is not None:
            mats = wp.to_torch(self.ball.root_view.get_material_properties()).clone()
            if self.cfg.ball_static_friction_range is not None:
                mu = sample(self.cfg.ball_static_friction_range, n)[:, None]
                mats[..., 0] = mu
                mats[..., 1] = mu * self.cfg.ball_dynamic_friction_ratio
            if self.cfg.ball_restitution_range is not None:
                mats[..., 2] = sample(self.cfg.ball_restitution_range, n)[:, None]
            self.ball.root_view.set_material_properties(wp.from_torch(mats, dtype=wp.float32), wp.from_torch(env_ids))
        if self.cfg.robot_friction_scale_range is not None:
            mats = wp.to_torch(self.robot.root_view.get_material_properties()).clone()
            scale = sample(self.cfg.robot_friction_scale_range, n)[:, None]
            mats[..., 0] *= scale
            mats[..., 1] *= scale
            self.robot.root_view.set_material_properties(wp.from_torch(mats, dtype=wp.float32), wp.from_torch(env_ids))

    def _set_real_arm_inertia(self):
        """Overwrite upper-arm and lower-arm (stick) mass, CoM and inertia with the real hardware values."""
        cfg, r = self.cfg, self.robot
        up_idx = r.find_bodies("upperarm")[0][0]
        la_idx = self._lowerarm_body_idx
        pos_w, quat_w = r.data.body_pos_w.torch[0], r.data.body_quat_w.torch[0]
        elbow_l = torch.tensor(STICK_ELBOW_OFFSET, device=self.device)
        lip_l = torch.tensor(STICK_LIP_OFFSET, device=self.device)
        elbow_w = pos_w[la_idx] + quat_apply(quat_w[la_idx], elbow_l)
        # the shoulder axis passes through the origin of the shoulder cylinder body
        shoulder_u = quat_apply_inverse(quat_w[up_idx], pos_w[self._shoulder_body_idx] - pos_w[up_idx])
        elbow_u = quat_apply_inverse(quat_w[up_idx], elbow_w - pos_w[up_idx])
        axis_u = (elbow_u - shoulder_u) / torch.linalg.norm(elbow_u - shoulder_u)
        com_u = shoulder_u + cfg.upperarm_com_from_shoulder * axis_u
        axis_l = (lip_l - elbow_l) / torch.linalg.norm(lip_l - elbow_l)
        com_l = elbow_l + cfg.stick_com_from_elbow * axis_l

        def rod_inertia(axis, perp, axial):
            eye = torch.eye(3, device=self.device)
            return (perp * eye + (axial - perp) * torch.outer(axis, axis)).reshape(9)

        stick_perp = cfg.stick_mass * cfg.stick_length**2 / 12.0
        body_ids = [up_idx, la_idx]
        n = self.num_envs
        masses = torch.tensor([cfg.upperarm_mass, cfg.stick_mass], device=self.device).repeat(n, 1)
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # (x, y, z, w)
        coms = torch.stack([torch.cat([com_u, ident]), torch.cat([com_l, ident])]).repeat(n, 1, 1)
        inertias = torch.stack(
            [
                rod_inertia(axis_u, cfg.upperarm_inertia_perp, cfg.upperarm_inertia_axial),
                rod_inertia(axis_l, stick_perp, cfg.stick_inertia_axial),
            ]
        ).repeat(n, 1, 1)
        r.set_masses_index(masses=masses, body_ids=body_ids)
        r.set_coms_index(coms=coms, body_ids=body_ids)
        r.set_inertias_index(inertias=inertias, body_ids=body_ids)

    def _setup_scene(self):
        # exact (custom-geometry) cylinder colliders for the wheels; the default convex approximation is a polygon
        # that rolls bumpily
        import carb

        carb.settings.get_settings().set_bool("/physics/collisionCylinderCustomGeometry", True)
        self.robot = Articulation(self.cfg.robot_cfg)
        if self.cfg.ball_shape == "ellipsoid":
            sp = self.cfg.ball_cfg.spawn
            self.cfg.ball_cfg = self.cfg.ball_cfg.replace(
                spawn=sim_utils.MeshSphereCfg(
                    radius=sp.radius,
                    rigid_props=sp.rigid_props,
                    mass_props=sp.mass_props,
                    collision_props=sp.collision_props,
                    physics_material=sp.physics_material,
                    visual_material=sp.visual_material,
                    activate_contact_sensors=True,
                )
            )
        self.ball = RigidObject(self.cfg.ball_cfg)
        if self.cfg.ball_shape == "ellipsoid":
            self._make_ball_ellipsoid("/World/envs/env_0/Ball")
        self.ball_contact = ContactSensor(self.cfg.ball_contact_sensor)

        self._use_floor = self.cfg.floor_tilt_max_deg > 0.0
        if self._use_floor:
            size = self.cfg.floor_size
            floor_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Floor",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=self.cfg.floor_friction, dynamic_friction=self.cfg.floor_friction
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.22, 0.24)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(self.cfg.floor_center_x, 0.0, -0.5 * size[2])),
            )
            self.floor = RigidObject(floor_cfg)
            # keep the global ground out of the way, below the platforms
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.2))
        else:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        if self._use_floor:
            self.scene.rigid_objects["floor"] = self.floor
        self.scene.sensors["ball_contact"] = self.ball_contact

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _make_ball_ellipsoid(self, prim_path: str):
        """Scale the (mesh) ball into a slight ellipsoid and collide with a fine convex hull instead of a sphere."""
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        scale_ops = [op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        scale = scale_ops[0] if scale_ops else xf.AddScaleOp()
        scale.Set(tuple(float(v) for v in self.cfg.ball_ellipsoid_scale))
        # replace the UV-sphere mesh (vertices crowd at the poles; the 255-vertex hull PhysX cooks from it is an
        # irregular polyhedron sitting ~3 mm low) by an icosphere: 162 evenly spread vertices, max inset ~0.6 mm,
        # so the convex hull is the mesh itself
        import trimesh

        ico = trimesh.creation.icosphere(subdivisions=2, radius=self.cfg.ball_radius)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                mesh.GetPointsAttr().Set([tuple(float(c) for c in v) for v in ico.vertices])
                mesh.GetFaceVertexCountsAttr().Set([3] * len(ico.faces))
                mesh.GetFaceVertexIndicesAttr().Set([int(i) for i in ico.faces.flatten()])
                UsdPhysics.MeshCollisionAPI.Apply(child).GetApproximationAttr().Set("convexHull")
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(child).CreateHullVertexLimitAttr(255)

    def _floor_height(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World z of the floor surface under the given world positions (one per env)."""
        rel = pos_w - self.scene.env_origins
        n = self._floor_normal
        return self.scene.env_origins[:, 2] - (n[:, 0] * rel[:, 0] + n[:, 1] * rel[:, 1]) / n[:, 2]

    def place_ball(self, env_ids: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """Put the ball on the floor at env-local (x, y); an imperfect ball rests on its short axis with random yaw."""
        origins = self.scene.env_origins[env_ids]
        n = self._floor_normal[env_ids]
        surf_z = -(n[:, 0] * x + n[:, 1] * y) / n[:, 2]
        pose = self.ball.data.default_root_pose.torch[env_ids].clone()
        pose[:, 0] = origins[:, 0] + x + n[:, 0] * self.cfg.ball_radius
        pose[:, 1] = origins[:, 1] + y + n[:, 1] * self.cfg.ball_radius
        pose[:, 2] = origins[:, 2] + surf_z + n[:, 2] * self.cfg.ball_radius * max(self.cfg.ball_ellipsoid_scale)
        # rest the ball the way a real one settles: an ellipsoid on its short axis (body z, scaled 0.98), an
        # unbalanced ball with its CoM straight below the center; then a random yaw about the vertical
        yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
        q_yaw = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        com = self._ball_com_offset[env_ids]
        has_com = torch.linalg.norm(com, dim=-1) > 1e-6
        if self.cfg.ball_shape != "ellipsoid" and has_com.any():
            # rotation taking the body CoM direction onto -z (world): axis = c x (-z), angle = acos(c . -z)
            c = com / torch.linalg.norm(com, dim=-1, keepdim=True).clamp(min=1e-9)
            down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(c)
            axis = torch.cross(c, down, dim=-1)
            sin_a = torch.linalg.norm(axis, dim=-1, keepdim=True)
            cos_a = (c * down).sum(-1, keepdim=True)
            half = 0.5 * torch.atan2(sin_a, cos_a)
            axis = torch.where(
                sin_a > 1e-6, axis / sin_a.clamp(min=1e-9), torch.tensor([1.0, 0.0, 0.0], device=self.device)
            )
            q_down = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)  # (x, y, z, w)
            from isaaclab.utils.math import quat_mul

            q = torch.where(has_com[:, None], quat_mul(q_yaw, q_down), q_yaw)
        else:
            q = q_yaw
        pose[:, 3:7] = q
        self.ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        self._ball_spawn_pos[env_ids] = pose[:, :3] - origins

    def _pocket_pos_w(self) -> torch.Tensor:
        lowerarm_pos_w = self.robot.data.body_pos_w.torch[:, self._lowerarm_body_idx]
        lowerarm_quat_w = self.robot.data.body_quat_w.torch[:, self._lowerarm_body_idx]
        offset = self._pocket_offset.expand(self.num_envs, -1)
        return lowerarm_pos_w + quat_apply(lowerarm_quat_w, offset)

    def _lip_pos_w(self) -> torch.Tensor:
        lowerarm_pos_w = self.robot.data.body_pos_w.torch[:, self._lowerarm_body_idx]
        lowerarm_quat_w = self.robot.data.body_quat_w.torch[:, self._lowerarm_body_idx]
        offset = self._lip_offset.expand(self.num_envs, -1)
        return lowerarm_pos_w + quat_apply(lowerarm_quat_w, offset)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions[:] = self._actions
        actions = actions.clone().clamp(-1.0, 1.0)
        if self._max_delay > 0:
            # control latency: the action applied now was issued `delay` policy steps ago
            self._action_hist = torch.roll(self._action_hist, shifts=1, dims=1)
            self._action_hist[:, 0] = actions
            actions = self._action_hist[torch.arange(self.num_envs, device=self.device), self._action_delay]
        self._actions = actions
        if self.cfg.arm_monotonic and not self.cfg.arm_locked:
            arm_upper = torch.tensor(self.cfg.arm_joint_target_upper, device=self.device)
            step = self._actions[:, 3:5].clamp(min=0.0) * self.cfg.arm_increment_max
            new_target = torch.minimum(self._arm_target + step, arm_upper)
            self._arm_vel_target = (new_target - self._arm_target) / self.step_dt
            self._arm_target = new_target

    def _apply_action(self) -> None:
        # base (vx, vy, wz) command in the chassis frame. With `base_drive == "wheels"` it becomes wheel velocity
        # targets (skid steer, vy ignored); the legacy "root_velocity" mode writes it straight to the root body.
        cmd_scale = torch.tensor(
            [self.cfg.base_lin_vel_scale, self.cfg.base_lin_vel_scale, self.cfg.base_ang_vel_scale], device=self.device
        )
        cmd = self._actions[:, :3].clone()
        if self.cfg.base_forward_only:
            cmd[:, 0] = 0.5 * (cmd[:, 0] + 1.0)
        cmd_b = cmd * cmd_scale
        # slip: the executed velocity lags behind a scaled, biased, noisy version of the command
        target_b = cmd_b * self._base_cmd_scale + self._base_cmd_bias
        target_b = target_b + torch.randn_like(target_b) * torch.tensor(
            self.cfg.base_slip_noise_std, device=self.device
        )
        delta = self._base_lag_alpha * (target_b - self._base_vel_b)
        # acceleration limits on the executed velocity
        dt = self.cfg.sim.dt
        if self.cfg.base_max_lin_accel is not None:
            lin = delta[:, :2]
            max_dv = self.cfg.base_max_lin_accel * dt
            norm = torch.linalg.norm(lin, dim=-1, keepdim=True).clamp(min=1e-9)
            delta[:, :2] = lin * torch.clamp(max_dv / norm, max=1.0)
        if self.cfg.base_max_ang_accel is not None:
            max_dw = self.cfg.base_max_ang_accel * dt
            delta[:, 2] = delta[:, 2].clamp(-max_dw, max_dw)
        self._base_vel_b += delta
        vx, vy, wz = self._base_vel_b[:, 0], self._base_vel_b[:, 1], self._base_vel_b[:, 2]

        if self.cfg.base_drive == "wheels":
            # skid steer: each wheel's ground speed is vx - wz * y_wheel; vy cannot be produced by cylinder wheels
            wheel_speed = (vx[:, None] - wz[:, None] * self._wheel_y[None, :]) / self.cfg.wheel_radius
            self.robot.set_joint_velocity_target_index(target=wheel_speed, joint_ids=self._wheel_joint_idx)
        else:
            self._write_root_velocity(vx, vy, wz)
        self._apply_arm_action()

    def _write_root_velocity(self, vx: torch.Tensor, vy: torch.Tensor, wz: torch.Tensor) -> None:
        root_quat_w = self.robot.data.root_quat_w.torch
        lin_vel_b = torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)
        lin_vel_w = quat_apply(root_quat_w, lin_vel_b)
        # only the planar (vx, vy, wz) components are commanded; keep the simulated vertical velocity
        # and roll/pitch rates so gravity and ground contact still act on the base
        lin_vel_w[:, 2] = self.robot.data.root_lin_vel_w.torch[:, 2]
        ang_vel_w = self.robot.data.root_ang_vel_w.torch.clone()
        ang_vel_w[:, 2] = wz
        root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
        self.robot.write_root_velocity_to_sim(root_vel)

    def _apply_arm_action(self) -> None:
        default = self._default_arm_joint_pos
        if self.cfg.arm_locked:
            arm_target = default
        elif self.cfg.arm_monotonic:
            arm_target = self._arm_target
        else:
            arm_action = self._actions[:, 3:5]
            arm_lower = torch.tensor(self.cfg.arm_joint_target_lower, device=self.device)
            arm_upper = torch.tensor(self.cfg.arm_joint_target_upper, device=self.device)
            arm_target = torch.where(
                arm_action < 0.0,
                default + arm_action * (default - arm_lower),
                default + arm_action * (arm_upper - default),
            )
        self.robot.set_joint_position_target_index(target=arm_target, joint_ids=self._arm_joint_idx)
        arm_vel_target = self._arm_vel_target if (self.cfg.arm_monotonic and not self.cfg.arm_locked) else None
        if arm_vel_target is not None:
            self.robot.set_joint_velocity_target_index(target=arm_vel_target, joint_ids=self._arm_joint_idx)
        arm_ff = self._arm_ff_extra.clone()
        if self.cfg.arm_gravity_comp:
            gc = self.robot.data.gravity_compensation_forces.torch
            offset = gc.shape[1] - self.robot.num_joints  # floating base: 6 root dofs come first
            arm_ff += self._gc_scale * gc[:, [offset + i for i in self._arm_joint_idx]]
        # joint friction (zero unless randomized)
        qd = self.robot.data.joint_vel.torch[:, self._arm_joint_idx]
        arm_ff -= self._arm_fric_coulomb * torch.tanh(qd / 0.02) + self._arm_fric_viscous * qd
        self.robot.set_joint_effort_target_index(target=arm_ff, joint_ids=self._arm_joint_idx)

    def _get_observations(self) -> dict:
        root_lin_vel_b = self.robot.data.root_lin_vel_b.torch
        root_ang_vel_b = self.robot.data.root_ang_vel_b.torch
        arm_pos = self.robot.data.joint_pos.torch[:, self._arm_joint_idx]
        arm_vel = self.robot.data.joint_vel.torch[:, self._arm_joint_idx]

        pocket_pos_w = self._pocket_pos_w()
        ball_pos_w = self.ball.data.root_pos_w.torch
        ball_vel_w = self.ball.data.root_lin_vel_w.torch

        root_quat_w = self.robot.data.root_quat_w.torch
        pocket_to_ball_b = quat_rotate_inverse(root_quat_w, ball_pos_w - pocket_pos_w)
        ball_vel_b = quat_rotate_inverse(root_quat_w, ball_vel_w)

        obs = torch.cat(
            (
                root_lin_vel_b,
                root_ang_vel_b,
                arm_pos,
                arm_vel,
                pocket_to_ball_b,
                ball_vel_b,
                self._actions,
            ),
            dim=-1,
        )
        # envs that blew up this step are reset in _get_dones, but their buffers can stay stale
        # until the next physics step -- keep those values out of the policy input.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pocket_pos_w = self._pocket_pos_w()
        ball_pos_w = self.ball.data.root_pos_w.torch

        ball_bottom_w = ball_pos_w.clone()
        ball_bottom_w[:, 2] -= self.cfg.ball_radius
        lip_dist = torch.linalg.norm(ball_bottom_w - self._lip_pos_w(), dim=-1)
        reach_reward = 1.0 - torch.tanh(lip_dist / self.cfg.reach_std)

        # height above the floor surface under the ball (the floor may be tilted)
        ball_height = ball_pos_w[:, 2] - self._floor_height(ball_pos_w)
        horiz_dist = torch.linalg.norm((ball_pos_w - pocket_pos_w)[:, :2], dim=-1)
        contained = horiz_dist < self.cfg.lift_containment_radius
        # ball <-> stick contact over the last env step (any physics substep), shape (N, 1, 1, 3) per substep
        net_force = self.ball_contact.data.force_matrix_w_history.torch
        net_contact = torch.linalg.norm(net_force, dim=-1).amax(dim=(1, 2, 3)) > self.cfg.net_contact_force_threshold
        clearance = ball_height - self.cfg.ball_radius
        off_ground = clearance > self.cfg.ground_contact_tolerance
        self._ball_in_net = net_contact & off_ground & contained
        if self.cfg.reward_stage == 1:
            in_net = (off_ground & contained).float()
        else:
            in_net = self._ball_in_net.float()
        reach_reward = torch.where(in_net > 0, torch.ones_like(reach_reward), reach_reward)
        # lift quality: little forward travel of the ball, low robot speed (the robot starts facing +x)
        ball_forward = (ball_pos_w[:, 0] - self.scene.env_origins[:, 0] - self._ball_spawn_pos[:, 0]).clamp(min=0.0)
        robot_speed = torch.linalg.norm(self.robot.data.root_lin_vel_w.torch[:, :2], dim=-1)
        quality = torch.exp(-ball_forward / self.cfg.pop_forward_scale) * torch.exp(
            -torch.square(robot_speed / self.cfg.pop_robot_speed_scale)
        )
        lifted = ball_height > self.cfg.lift_height_threshold
        lift_reward = in_net * lifted.float() * quality
        # dense shaping: reward a ball in the net for how high it has risen, saturating at ~0.1 m clearance
        raise_reward = in_net * torch.tanh(clearance.clamp(min=0.0) / self.cfg.raise_std) * quality

        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=-1)

        total_reward = (
            self.cfg.rew_scale_reach * reach_reward
            + self.cfg.rew_scale_lift * lift_reward
            + self.cfg.rew_scale_raise * raise_reward
            + self.cfg.rew_scale_action_rate * action_rate
            + self.cfg.rew_scale_terminated * self.reset_terminated.float()
        )
        return torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ball_pos_local = self.ball.data.root_pos_w.torch - self.scene.env_origins
        ball_out_of_bounds = torch.linalg.norm(ball_pos_local[:, :2], dim=-1) > self.cfg.arena_bound

        projected_gravity_b = self.robot.data.projected_gravity_b.torch
        tipped_over = -projected_gravity_b[:, 2] < self.cfg.tilt_threshold

        # physics blow-ups (e.g. the stick's SDF colliders penetrating the ground) produce huge or
        # non-finite states; reset those envs instead of letting NaNs reach the policy.
        root_state = torch.cat((self.robot.data.root_pos_w.torch, self.robot.data.root_lin_vel_w.torch), dim=-1)
        ball_state = torch.cat((self.ball.data.root_pos_w.torch, self.ball.data.root_lin_vel_w.torch), dim=-1)
        joint_state = torch.cat((self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch), dim=-1)
        non_finite = ~(
            torch.isfinite(root_state).all(dim=-1)
            & torch.isfinite(ball_state).all(dim=-1)
            & torch.isfinite(joint_state).all(dim=-1)
        )
        too_fast = (
            torch.linalg.norm(torch.nan_to_num(ball_state[:, 3:], nan=0.0), dim=-1) > self.cfg.max_ball_speed
        ) | (torch.linalg.norm(torch.nan_to_num(root_state[:, 3:], nan=0.0), dim=-1) > self.cfg.max_robot_speed)

        terminated = ball_out_of_bounds | tipped_over | non_finite | too_fast
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        self._prev_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._arm_target[env_ids] = self._default_arm_joint_pos[env_ids]
        self._arm_vel_target[env_ids] = 0.0
        # per-episode uncertainty samples
        cfg = self.cfg
        shape = (num_resets, 2)
        self._arm_fric_coulomb[env_ids] = sample_uniform(*cfg.arm_friction_coulomb_range, shape, self.device)
        self._arm_fric_viscous[env_ids] = sample_uniform(*cfg.arm_friction_viscous_range, shape, self.device)
        self._gc_scale[env_ids] = sample_uniform(*cfg.arm_gravity_comp_scale_range, shape, self.device)
        lo, hi = cfg.action_delay_steps_range
        self._action_delay[env_ids] = torch.randint(int(lo), int(hi) + 1, (num_resets,), device=self.device)
        self._action_hist[env_ids] = 0.0

        # resample base slip parameters for the new episode
        self._base_cmd_scale[env_ids] = sample_uniform(
            self.cfg.base_slip_scale_range[0], self.cfg.base_slip_scale_range[1], (num_resets, 3), self.device
        )
        bias_max = torch.tensor(self.cfg.base_slip_bias_max, device=self.device)
        self._base_cmd_bias[env_ids] = sample_uniform(-1.0, 1.0, (num_resets, 3), self.device) * bias_max
        tau = sample_uniform(
            self.cfg.base_slip_lag_tau_range[0], self.cfg.base_slip_lag_tau_range[1], (num_resets, 1), self.device
        )
        self._base_lag_alpha[env_ids] = self.cfg.sim.dt / (tau + self.cfg.sim.dt)
        # the robot starts already rolling forward toward the ball
        init_vel_b = torch.stack(
            [
                sample_uniform(r[0], r[1], (num_resets,), self.device)
                for r in (
                    self.cfg.base_init_vel_x_range,
                    self.cfg.base_init_vel_y_range,
                    self.cfg.base_init_yaw_rate_range,
                )
            ],
            dim=-1,
        )
        self._base_vel_b[env_ids] = init_vel_b

        # reset robot to its default state
        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        root_vel = torch.zeros(num_resets, 6, device=self.device)
        root_vel[:, :3] = quat_apply(default_root_pose[:, 3:7], torch.cat((init_vel_b[:, :2], root_vel[:, :1]), dim=-1))
        root_vel[:, 5] = init_vel_b[:, 2]
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)

        # floor that is flat but not level: random roll/pitch about the robot's start point
        if self._use_floor:
            m = math.radians(self.cfg.floor_tilt_max_deg)
            roll = sample_uniform(-m, m, (num_resets,), self.device)
            pitch = sample_uniform(-m, m, (num_resets,), self.device)
            q = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
            up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(num_resets, -1)
            self._floor_normal[env_ids] = quat_apply(q, up)
            center = torch.tensor([[self.cfg.floor_center_x, 0.0, -0.5 * self.cfg.floor_size[2]]], device=self.device)
            floor_pose = torch.cat(
                [self.scene.env_origins[env_ids] + quat_apply(q, center.expand(num_resets, -1)), q], dim=-1
            )
            self.floor.write_root_pose_to_sim_index(root_pose=floor_pose, env_ids=env_ids)

        # randomize ball position on the floor, in front of the robot
        x = sample_uniform(self.cfg.ball_spawn_x_range[0], self.cfg.ball_spawn_x_range[1], (num_resets,), self.device)
        y = sample_uniform(self.cfg.ball_spawn_y_range[0], self.cfg.ball_spawn_y_range[1], (num_resets,), self.device)
        self.place_ball(env_ids, x, y)
