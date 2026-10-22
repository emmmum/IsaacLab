# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the retargeted scoop on a Franka Research 3 in simulation (physics, ball, floor).

The FR3 (``fr3_lacrosse.usd``: stick welded to the hand at the planned grip) tracks the planned joint trajectory from
``plan_fr3.py`` with stiff joint PD + velocity feed-forward (gravity off on the arm, as with the FR3's own gravity
compensation). Each env gets the ball at the planned spot plus a random offset; success = at the end the ball is more
than 15 cm above the floor and within 15 cm of the pocket.

Usage:
  ./isaaclab.sh -p logs/scoop_traj_search/fr3/fr3_scoop_sim.py --plan logs/scoop_traj_search/fr3/fr3_plan.npz \
      [--num_envs 64 --ball_offset 0.01 --video out.mp4]
"""

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--ball_offset", type=float, default=0.01, help="uniform +- ball offset in the scoop frame [m]")
p.add_argument("--video", default="", help="mp4 of env 0 (needs --enable_cameras)")
p.add_argument("--video_every", type=int, default=6, help="physics steps per frame (6 = real time at 60 fps)")
p.add_argument("--out", default="", help="save per-step signals of env 0 to this .npz")
p.add_argument("--usd", default="fr3_lacrosse.usd", help="FR3 + stick asset (in the robomaster_lacrosse data dir)")
p.add_argument("--husky", action="store_true", help="put the FR3 on a Clearpath Husky (mount_x from the plan)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
USD = os.path.join(REPO, "source/isaaclab_assets/data/robomaster_lacrosse", a.usd)
HUSKY_USD = os.path.join(REPO, "source/isaaclab_assets/data/husky/usd/husky/husky.usda")
HUSKY_BASE_Z = 0.1323  # [m] base_link above the floor (Clearpath husky_description); top plate at 0.3773 m
BALL_R, BALL_M = 0.0325, 0.141
POCKET = (0.46, 0.016, 0.46)  # [m] pocket in the stick frame (STICK_POCKET_OFFSET)
LIP = (0.640, 0.016, 0.452)  # [m] lip in the stick frame (STICK_LIP_OFFSET)
ROLL_RES, ROLL_W0 = 0.015, 0.05  # rolling resistance, as in the RoboMaster env

plan = np.load(a.plan)
q_plan, qd_plan, t_plan, dt_plan = plan["q"], plan["qd"], plan["t"], float(plan["dt"])
base_h = float(plan["base_h"])
T_ball = plan["T_ball_in_base"]  # ball (floor) frame in the FR3 base frame
q0 = q_plan[0]
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # FR3 on a Husky (top plate at the FR3 base, FR3 mount_x ahead of the Husky center), or on a plain stand
    husky = (
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Husky",
            spawn=sim_utils.UsdFileCfg(usd_path=HUSKY_USD),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-float(plan["mount_x"]), 0.0, HUSKY_BASE_Z)),
        )
        if a.husky
        else None
    )
    stand = None if a.husky else AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, max(base_h, 0.01)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * base_h)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_h),
            joint_pos={**{j: float(v) for j, v in zip(JOINTS, q0)}, "fr3_finger_joint.*": 0.04},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"], stiffness=4000.0, damping=150.0, effort_limit_sim=200.0,
                velocity_limit_sim=6.0,
            ),
            "fingers": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], stiffness=2000.0, damping=100.0),
        },
    )  # fmt: skip
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_R,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_M),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_R)),
    )


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 360, render_interval=6, physics=PhysxCfg(gpu_collision_stack_size=2**29))
    )
    scene = InteractiveScene(SceneCfg(num_envs=a.num_envs, env_spacing=3.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    ball: RigidObject = scene["ball"]
    dev, n = sim.device, a.num_envs
    o = scene.env_origins
    arm_ids = robot.find_joints(JOINTS, preserve_order=True)[0]
    stick_id = robot.find_bodies("stick")[0][0]
    hand_id = robot.find_bodies("fr3_hand")[0][0]

    # start the arm at the first planned configuration (fingers open) and hold it there
    jp = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=jp)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
    robot.set_joint_position_target_index(target=jp)
    # ball at the planned spot (+ offset in the scoop frame), at rest on the floor
    Tb = torch.tensor(T_ball, dtype=torch.float32, device=dev)
    off = (torch.rand(n, 2, device=dev) * 2 - 1) * a.ball_offset
    off[0] = 0.0
    off3 = torch.cat([off, torch.zeros(n, 1, device=dev)], -1)
    ball_xy = Tb[:3, 3] + off3 @ Tb[:3, :3].T
    pose = ball.data.default_root_pose.torch.clone()
    pose[:, :3] = o + ball_xy + torch.tensor([0.0, 0.0, base_h + BALL_R], device=dev)
    ball.write_root_pose_to_sim_index(root_pose=pose)
    ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    # FK check: hand pose vs the planner's TCP target at t = 0 (TCP = hand + 0.1034 m along hand z)
    hand = robot.data.body_pose_w.torch[0, hand_id]
    tcp = hand[:3] + quat_apply(hand[3:7][None], torch.tensor([[0.0, 0.0, 0.1034]], device=dev))[0] - o[0]
    T_tcp0 = np.asarray(plan["T_ball_in_base"]) @ plan["T_stick_ball"][0] @ plan["T_tcp_in_stick"]
    print(f"SIM FK check: TCP sim {(tcp.cpu().numpy() - [0, 0, base_h]).round(4)} vs plan {T_tcp0[:3, 3].round(4)}")

    phys_dt = sim.get_physics_dt()
    steps = int(round((t_plan[-1] - t_plan[0]) / phys_dt))
    q_t = torch.tensor(q_plan, dtype=torch.float32, device=dev)
    qd_t = torch.tensor(qd_plan, dtype=torch.float32, device=dev)
    lip_off = torch.tensor([LIP], device=dev).expand(n, -1)
    pocket_off = torch.tensor([POCKET], device=dev).expand(n, -1)
    rec = {k: [] for k in ("t", "q", "q_ref", "lip", "ball")}
    writer = None
    if a.video:
        import imageio

        import omni.replicator.core as rep

        writer = imageio.get_writer(a.video, fps=60, codec="libx264", quality=8, macro_block_size=8)
        # side view of the scoop line: look at the middle between the FR3 base and the ball, from the side
        mid = (o[0] + 0.5 * torch.tensor([float(ball_xy[0, 0]), float(ball_xy[0, 1]), 0.0], device=dev)).tolist()
        head = Tb[:3, 0].tolist()  # scoop direction
        side = [-head[1], head[0]]
        from pxr import Gf, UsdGeom

        from isaaclab.sim.utils import get_current_stage

        eye = Gf.Vec3d(mid[0] + 2.2 * side[0] - 0.6 * head[0], mid[1] + 2.2 * side[1] - 0.6 * head[1], 1.0)
        cam = UsdGeom.Camera.Define(get_current_stage(), "/World/VideoCam")
        cam.CreateFocalLengthAttr(18.0)
        view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(mid[0], mid[1], 0.35), Gf.Vec3d(0, 0, 1))
        cam.AddTransformOp().Set(view.GetInverse())
        rp = rep.create.render_product("/World/VideoCam", (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
        for _ in range(10):
            sim.render()

    for k in range(steps):
        x = min(k * phys_dt / dt_plan, len(t_plan) - 1.001)
        i, f = int(x), x - int(x)
        q_ref = q_t[i] * (1 - f) + q_t[i + 1] * f
        qd_ref = qd_t[i] * (1 - f) + qd_t[i + 1] * f
        robot.set_joint_position_target_index(target=q_ref.expand(n, -1), joint_ids=arm_ids)
        robot.set_joint_velocity_target_index(target=qd_ref.expand(n, -1), joint_ids=arm_ids)
        # ball rolling resistance while on the floor
        bp = ball.data.root_pos_w.torch
        on_floor = (bp[:, 2] - o[:, 2] - BALL_R) < 0.002  # the floor is at the env origin height
        w = ball.data.root_ang_vel_w.torch.clone()
        w[:, 2] = 0.0
        wn = torch.linalg.norm(w, dim=-1, keepdim=True)
        tq = -ROLL_RES * BALL_M * 9.81 * BALL_R * torch.tanh(wn / ROLL_W0) * w / wn.clamp(min=1e-9)
        ball.instantaneous_wrench_composer.add_forces_and_torques_index(
            torques=(tq * on_floor[:, None].float())[:, None], body_ids=torch.tensor([0], dtype=torch.int32, device=dev),
            is_global=True,
        )  # fmt: skip
        scene.write_data_to_sim()
        render = writer is not None and k % a.video_every == 0
        sim.step(render=render)
        scene.update(phys_dt)
        sp = robot.data.body_pose_w.torch[:, stick_id]
        if a.out and k % 3 == 0:
            rec["t"].append(t_plan[0] + k * phys_dt)
            rec["q"].append(robot.data.joint_pos.torch[0, arm_ids].cpu().numpy())
            rec["q_ref"].append(q_ref.cpu().numpy())
            rec["lip"].append((sp[0, :3] + quat_apply(sp[:1, 3:7], lip_off[:1])[0] - o[0]).cpu().numpy())
            rec["ball"].append((bp[0] - o[0]).cpu().numpy())
        if render:
            frame = np.asarray(rgb.get_data())
            if frame.size and frame.mean() > 5:
                writer.append_data(frame[..., :3])
    sp = robot.data.body_pose_w.torch[:, stick_id]
    pocket = sp[:, :3] + quat_apply(sp[:, 3:7], pocket_off)
    bp = ball.data.root_pos_w.torch
    height = bp[:, 2] - o[:, 2] - BALL_R
    near = torch.linalg.norm(bp - pocket, dim=-1) < 0.15
    ok = (height > 0.15) & near
    err = (robot.data.joint_pos.torch[:, arm_ids] - q_t[-1]).abs().max().item()
    print(f"SIM success {int(ok.sum())}/{n} (ball offset +-{100 * a.ball_offset:.0f} cm); env0 ball height "
          f"{height[0]:.3f} m; final joint error {err:.4f} rad")  # fmt: skip
    if a.out:
        np.savez(a.out, **{k: np.array(v) for k, v in rec.items()})
    if writer is not None:
        writer.close()


main()
app.close()
