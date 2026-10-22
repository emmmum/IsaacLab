# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a self-contained package of the RoboMaster + 2-link arm + lacrosse stick assets.

* The robot USDs are flattened, the stray ``/FlatGrid`` environment prim (a remote reference saved with the robot) is
  removed, texture paths are made relative, and the real mass properties are baked in. In Isaac Lab those are only
  applied at runtime by the task (upper arm / stick / shoulder motor / chassis), so they are read back from a running
  environment and written as ``UsdPhysics.MassAPI`` on every rigid body.
* The stick alone is exported as one rigid body with its real mass properties.

Usage:
  ./isaaclab.sh -p source/isaaclab_assets/data/robomaster_lacrosse/export_package.py --out <dir>
"""

import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", required=True, help="output directory (its usd/ subfolder receives the assets)")
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli  # noqa: E402

add_launcher_args(parser)
args, rem = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + rem

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = "Isaac-RoboMaster-Lacrosse-Direct-v0"
ROBOTS = {
    # exported name: (source file, Isaac Lab env setting that matches it)
    "robomaster_lacrosse_wheeled.usd": ("robomaster_lacrosse_long_stick_wheeled.usd", "wheels"),
    "robomaster_lacrosse.usd": ("robomaster_lacrosse_long_stick.usd", "root_velocity"),
}


def read_mass_properties(env_cfg, base_drive):
    """Run the task once and return {body name: (mass, com pos, com quat (x,y,z,w), inertia 3x3)}."""
    env_cfg.scene.num_envs = 1
    env_cfg.base_drive = base_drive
    if base_drive != "wheels":
        from isaaclab_assets.robots.robomaster_lacrosse import ROBOMASTER_LACROSSE_LONG_STICK_CFG

        env_cfg.robot_cfg = ROBOMASTER_LACROSSE_LONG_STICK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    view = env.robot.root_view
    import warp as wp

    masses = wp.to_torch(view.get_masses()).reshape(-1).cpu().numpy()
    coms = wp.to_torch(view.get_coms()).reshape(len(masses), -1).cpu().numpy()
    inertias = wp.to_torch(view.get_inertias()).reshape(len(masses), 9).cpu().numpy()
    props = {}
    for i, name in enumerate(env.robot.body_names):
        props[name] = (float(masses[i]), coms[i, :3], coms[i, 3:7], inertias[i].reshape(3, 3))
    env.close()
    return props


def bake(src, dst, props):
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

    flat = Usd.Stage.Open(os.path.join(HERE, src)).Flatten()
    stage = Usd.Stage.Open(flat)
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        if prim != stage.GetDefaultPrim():
            stage.RemovePrim(prim.GetPath())  # e.g. /FlatGrid, saved together with the robot
    n_mass = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in props:
            m, com, q, inertia = props[prim.GetName()]
            ev, evec = np.linalg.eigh(inertia)
            if np.linalg.det(evec) < 0:
                evec[:, 0] *= -1
            rot = Gf.Matrix3d(*evec.T.flatten().tolist()).ExtractRotation().GetQuat()  # Gf uses row vectors
            # principal axes relative to the body frame: compose the PhysX CoM frame rotation with the eigenvectors
            q_com = Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            pa = q_com * rot
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(m)
            api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(x) for x in ev]))
            api.CreatePrincipalAxesAttr(Gf.Quatf(pa.GetReal(), *pa.GetImaginary()))
            api.CreateDensityAttr(0.0)
            n_mass += 1
    # textures next to the USD
    for prim in stage.Traverse():
        sh = UsdShade.Shader(prim)
        if sh:
            for inp in sh.GetInputs():
                v = inp.Get()
                if isinstance(v, Sdf.AssetPath) and v.path.endswith(".png") and "stick_textures" in v.path:
                    inp.Set(Sdf.AssetPath("./stick_textures/" + os.path.basename(v.path)))
    stage.GetRootLayer().Export(dst)
    print(f"[export] {dst}: mass properties on {n_mass} bodies")


def export_stick(dst, props):
    """The stick alone (the RoboMaster `lowerarm` frame: shaft along +x, elbow at x = 0) as one rigid body."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    src = Usd.Stage.Open(os.path.join(HERE, "robomaster_lacrosse_long_stick_wheeled.usd")).Flatten()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/lacrosse_stick").GetPrim()
    stage.SetDefaultPrim(root)
    layer = stage.GetRootLayer()
    nike = "Nike_Lunar_Elite_Womens_Lacrosse_Stick"
    for s, d in (
        (f"/robomaster/lowerarm/{nike}", f"/lacrosse_stick/{nike}"),
        ("/robomaster/stick_looks", "/lacrosse_stick/Looks"),
    ):
        Sdf.CopySpec(src, s, layer, d)
    for prim in Usd.PrimRange(root):
        for rel in prim.GetRelationships():
            t = [Sdf.Path(str(x).replace("/robomaster/stick_looks", "/lacrosse_stick/Looks")) for x in rel.GetTargets()]
            if t:
                rel.SetTargets(t)
        for attr in prim.GetAttributes():
            v = attr.Get()
            if isinstance(v, Sdf.AssetPath) and v.path.endswith(".png"):
                attr.Set(Sdf.AssetPath("./stick_textures/" + os.path.basename(v.path)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    m, com, q, inertia = props["lowerarm"]
    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(m)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(x) for x in np.linalg.eigvalsh(inertia)]))
    layer.Export(dst)
    print(f"[export] {dst}: stick {m:.3f} kg, CoM {np.round(com, 3)}")


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    with launch_simulation(env_cfg, args):
        out = os.path.join(os.path.abspath(args.out), "usd")
        os.makedirs(out, exist_ok=True)
        shutil.copytree(os.path.join(HERE, "stick_textures"), os.path.join(out, "stick_textures"), dirs_exist_ok=True)
        props_wheels = None
        for name, (src, drive) in ROBOTS.items():
            props = read_mass_properties(env_cfg.copy() if hasattr(env_cfg, "copy") else env_cfg, drive)
            for b, (m, com, _, inertia) in props.items():
                print(
                    f"[export] {name} {b}: {m:.3f} kg, CoM {np.round(com, 3)}, I diag {np.round(np.diag(inertia), 5)}"
                )
            bake(src, os.path.join(out, name), props)
            props_wheels = props_wheels or props
        export_stick(os.path.join(out, "lacrosse_stick.usd"), props_wheels)


if __name__ == "__main__":
    main()
