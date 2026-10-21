# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: an array of RoboMaster robots scooping lacrosse balls with the robust open-loop trajectory.

Opens the Isaac Sim window with an N x N grid of robots. Every round, each robot gets a ball about 1 m ahead (with a
small random offset, centered on the stick head, which sits 1.6 cm left of the robot's center line), drives at it, scoops it and holds it up; then all robots reset and the next round starts. The
number of robots holding their ball is printed after every round. Close the window or press Ctrl+C to stop.

All realistic uncertainty of the task config is on (base slip and acceleration limit, latency, contact parameters).

Usage:
  ./isaaclab.sh -p scripts/demos/robomaster_lacrosse_scoop.py
  ./isaaclab.sh -p scripts/demos/robomaster_lacrosse_scoop.py --grid 10 --speed 0.25   # 100 robots, 4x slow motion

  # record 3 rounds into an animated USD (no physics), then replay it in Isaac Sim with the timeline
  ./isaaclab.sh -p scripts/demos/robomaster_lacrosse_scoop.py --viz none --speed 100 --rounds 3 --save_usd anim.usd
  ./isaaclab.sh -p scripts/demos/play_usd_animation.py --usd anim.usd
"""

import argparse
import ctypes
import glob
import os
import site
import sys
import time

# torch (CUDA 13) JIT-compiles kernels with NVRTC, which dlopen()s libnvrtc-builtins by soname; the pip wheel's lib
# directory is not on the loader path in this environment, so load it up front (a no-op where it is not installed)
for _lib in sorted(
    glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "cu13", "lib", "libnvrtc-builtins.so.*"))
):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

# open the GUI unless the user picked a visualizer or headless mode
if not any(arg.startswith(("--viz", "--visualizer", "--headless")) for arg in sys.argv[1:]):
    sys.argv += ["--viz", "kit"]

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

parser = argparse.ArgumentParser(description="RoboMaster lacrosse scoop array demo.")
parser.add_argument("--task", default="Isaac-RoboMaster-Lacrosse-Direct-v0")
parser.add_argument("--grid", type=int, default=8, help="robots per side (grid x grid robots)")
parser.add_argument("--spacing", type=float, default=2.5, help="distance between robots [m]")
parser.add_argument("--speed", type=float, default=1.0, help="playback speed vs. real time (0.25 = 4x slow motion)")
parser.add_argument("--rounds", type=int, default=0, help="number of rounds to run (0 = until the window is closed)")
parser.add_argument("--ball_dx", type=float, default=0.03, help="uniform +- ball distance offset [m]")
parser.add_argument("--ball_dy", type=float, default=0.01, help="uniform +- ball lateral offset [m]")
parser.add_argument("--save_usd", default="", help="record the run into this animated USD (needs --rounds > 0)")
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining

# robust open-loop trajectory (logs/scoop_traj_search/best_trajectory_robust.json): drive at v from t = 0 with the
# elbow pressing the lip down, lift both joints at `rate` from t_lift by d_* above the start pose, stop driving
# t_brake after the lift
TRAJ = dict(v=0.9, t_lift=0.9, rate=1.0, d_shoulder=0.3, d_elbow=0.7, t_brake=1.0, press_elbow=-1.0)
BALL_X = 1.0  # [m] nominal ball distance ahead of the robot
SCOOP_S = 3.0  # [s] drive, scoop and brake
HOLD_S = 1.0  # [s] hold the ball up before the next round


def save_usd_animation(path: str, env, body_poses: list, ball_poses: list, camera: tuple):
    """Write the recorded run as a physics-free USD whose robot links and balls are animated with time samples.

    Each robot references the robot USD (so meshes and materials are shared, not copied); its rigid bodies get a
    world-space translate/orient animation, and its physics (body, collider and articulation schemas, joints) is
    removed so pressing Play only runs the animation.

    Args:
        path: Output ``.usd`` file.
        env: The environment the run was recorded in.
        body_poses: Per control step, link poses ``[pos, quat (x, y, z, w)]``, shape (num_envs, num_bodies, 7).
        ball_poses: Per control step, ball poses, shape (num_envs, 7).
        camera: ``(eye, target)`` of the overview camera [m].
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    from isaaclab.sim.spawners.from_files import GroundPlaneCfg

    path = os.path.abspath(path)
    bodies = torch.stack(body_poses).cpu().numpy()  # (T, N, B, 7)
    balls = torch.stack(ball_poses).cpu().numpy()  # (T, N, 7)
    n_frames, n_envs = bodies.shape[:2]
    robot_usd = os.path.relpath(env.cfg.robot_cfg.spawn.usd_path, os.path.dirname(path))

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(1.0 / env.step_dt)
    stage.SetFramesPerSecond(1.0 / env.step_dt)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_frames - 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.DefinePrim("/World/Ground").GetReferences().AddReference(GroundPlaneCfg().usd_path)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(2000.0)
    cam = UsdGeom.Camera.Define(stage, "/World/DemoCamera")
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera[0]), Gf.Vec3d(*camera[1]), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())

    ball_mat = UsdShade.Material.Define(stage, "/World/Looks/ball")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/ball/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.75, 0.1))
    ball_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    robots = []
    for e in range(n_envs):
        robot = stage.DefinePrim(f"/World/envs/env_{e}/Robot")
        robot.GetReferences().AddReference(robot_usd)
        robots.append(robot)
    # body prims and physics prims of the robot (the same relative paths in every copy)
    body_rel, physics = {}, []
    for prim in Usd.PrimRange(robots[0]):
        rel = prim.GetPath().MakeRelativePath(robots[0].GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_rel[prim.GetName()] = rel
        if any(schema.startswith(("Physics", "Physx")) for schema in prim.GetAppliedSchemas()):
            physics.append((rel, "api"))
        if prim.IsA(UsdPhysics.Joint):
            physics.append((rel, "joint"))
    scales = {
        name: robots[0].GetPrimAtPath(robots[0].GetPath().AppendPath(rel)).GetAttribute("xformOp:scale").Get()
        for name, rel in body_rel.items()
    }
    for e, robot in enumerate(robots):
        root = robot.GetPath()
        for rel, kind in physics:
            prim = stage.GetPrimAtPath(root.AppendPath(rel))
            if kind == "joint":
                prim.SetActive(False)
            else:
                # remove (not just disable) the physics schemas: otherwise PhysX still owns the body transforms
                # in Fabric during playback and the renderer ignores the time samples
                for schema in prim.GetAppliedSchemas():
                    if schema.startswith(("Physics", "Physx")):
                        prim.RemoveAppliedSchema(schema)
        for b, name in enumerate(env.robot.body_names):
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(root.AppendPath(body_rel[name])))
            ops = [xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "anim"), xf.AddOrientOp(opSuffix="anim")]
            if scales[name] is not None:
                ops.append(xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "anim"))
                ops[2].Set(Gf.Vec3f(scales[name]))
            xf.SetXformOpOrder(ops, resetXformStack=True)  # poses are recorded in the world frame
            for k in range(n_frames):
                p = bodies[k, e, b]
                ops[0].Set(Gf.Vec3d(*p[:3].tolist()), k)
                ops[1].Set(Gf.Quatf(float(p[6]), float(p[3]), float(p[4]), float(p[5])), k)
        ball = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{e}/Ball")
        ball.CreateRadiusAttr(env.cfg.ball_radius)
        UsdShade.MaterialBindingAPI.Apply(ball.GetPrim()).Bind(ball_mat)
        t_op, q_op = ball.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), ball.AddOrientOp()
        for k in range(n_frames):
            p = balls[k, e]
            t_op.Set(Gf.Vec3d(*p[:3].tolist()), k)
            q_op.Set(Gf.Quatf(float(p[6]), float(p[3]), float(p[4]), float(p[5])), k)
    stage.GetRootLayer().Save()
    print(f"[lacrosse demo] saved {n_frames} frames ({n_frames * env.step_dt:.1f} s) of {n_envs} robots to {path}")


def app_running() -> bool:
    """Whether the Isaac Sim window is still open."""
    import omni.kit.app

    return omni.kit.app.get_app().is_running()


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, args):
        n = args.grid * args.grid
        env_cfg.scene.num_envs = n
        env_cfg.scene.env_spacing = args.spacing
        env_cfg.episode_length_s = SCOOP_S + HOLD_S + 2.0  # rounds are reset by this script, not by time-out
        env_cfg.ball_spawn_x_range = (BALL_X, BALL_X)
        env_cfg.ball_spawn_y_range = (0.0, 0.0)
        env_cfg.arm_locked = False
        env_cfg.arm_monotonic = True
        env_cfg.base_forward_only = True
        if args.save_usd and args.rounds == 0:
            args.rounds = 3
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        dev, cfg = env.device, env.cfg

        # oblique camera over the whole grid
        env.reset()
        origins = env.scene.env_origins
        center = origins.mean(dim=0).tolist()
        span = args.grid * args.spacing
        camera = (
            (center[0] - 0.55 * span, center[1] - 0.72 * span, 0.42 * span),
            (center[0] + 0.8, center[1], 0.0),
        )
        env.sim.set_camera_view(eye=camera[0], target=camera[1])
        body_poses, ball_poses = [], []

        d_max = torch.tensor([[TRAJ["d_shoulder"], TRAJ["d_elbow"]]], device=dev)
        n_steps = int(round((SCOOP_S + HOLD_S) / env.step_dt))
        hold_steps = int(round(HOLD_S / env.step_dt))
        rnd, total_ok = 0, 0
        while app_running() and (args.rounds == 0 or rnd < args.rounds):
            env.reset()
            # ball about 1 m ahead of each robot, small random offset inside the alignment tolerance
            pose = env.ball.data.root_pose_w.torch.clone()
            pose[:, 0] = origins[:, 0] + BALL_X + (torch.rand(n, device=dev) * 2 - 1) * args.ball_dx
            pose[:, 1] = origins[:, 1] + cfg.ball_spawn_y_center + (torch.rand(n, device=dev) * 2 - 1) * args.ball_dy
            pose[:, 2] = origins[:, 2] + cfg.ball_radius
            env.ball.write_root_pose_to_sim_index(root_pose=pose)
            env._ball_spawn_pos[:] = pose[:, :3] - origins

            alive = torch.ones(n, dtype=torch.bool, device=dev)
            streak = torch.zeros(n, device=dev)
            wall0 = time.time()
            for k in range(n_steps):
                if not app_running():
                    break
                t = k * env.step_dt
                act = torch.zeros(n, 5, device=dev)
                driving = t < TRAJ["t_lift"] + TRAJ["t_brake"]
                act[:, 0] = (2.0 * TRAJ["v"] / cfg.base_lin_vel_scale - 1.0) if driving else -1.0
                if t >= TRAJ["t_lift"]:
                    remaining_lift = (d_max - (env._arm_target - env._default_arm_joint_pos)).clamp(min=0.0)
                    act[:, 3:5] = (remaining_lift.clamp(max=TRAJ["rate"] * env.step_dt) / cfg.arm_increment_max).clamp(
                        0, 1
                    )
                act[~alive, 0] = -1.0  # a robot whose episode ended (ball out of the arena) just waits
                act[~alive, 3:5] = 0.0
                env._arm_ff_extra[:, 1] = TRAJ["press_elbow"] if t < TRAJ["t_lift"] else 0.0
                _, _, term, trunc, _ = env.step(act)
                alive &= ~(term | trunc)
                streak = torch.where(env._ball_in_net & alive, streak + 1, torch.zeros_like(streak))
                if args.save_usd:
                    body_poses.append(env.robot.data.body_link_pose_w.torch.clone())
                    ball_poses.append(env.ball.data.root_pose_w.torch.clone())
                # pace the simulation to real time (scaled by --speed)
                lag = (k + 1) * env.step_dt / args.speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
            ok = int(((streak >= hold_steps) & alive).sum())
            rnd += 1
            total_ok += ok
            print(
                f"[lacrosse demo] round {rnd}: {ok}/{n} robots holding the ball"
                f" ({100.0 * total_ok / (rnd * n):.1f}% overall)"
            )
        if args.save_usd and body_poses:
            save_usd_animation(args.save_usd, env, body_poses, ball_poses, camera)
        env.close()


if __name__ == "__main__":
    main()
