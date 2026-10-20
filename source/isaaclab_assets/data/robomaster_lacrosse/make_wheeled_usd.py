# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the wheel-driven RoboMaster-lacrosse USD from the long-stick asset.

Changes relative to ``robomaster_lacrosse_long_stick.usd``:

* the articulation root moves from the top-level Xform (PhysX picked the shoulder cylinder as root link) to the
  chassis body ``base_link/base``;
* the mecanum-wheel SDF mesh colliders (rigid, non-rolling rollers) and the bolt colliders are disabled, and each
  wheel link gets a plain cylinder collider (radius/width from the wheel mesh) with a rubber-like material;
* the four wheel joints get an angular drive so they can be velocity-controlled;
* each wheel link gets an explicit mass (the original 0.232 kg).

Usage: ./isaaclab.sh -p source/isaaclab_assets/data/robomaster_lacrosse/make_wheeled_usd.py
"""

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import os  # noqa: E402

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "robomaster_lacrosse_long_stick.usd")
DST = os.path.join(HERE, "robomaster_lacrosse_long_stick_wheeled.usd")
WHEELS = ("rear_left", "rear_right", "front_left", "front_right")
WHEEL_STATIC_FRICTION = 1.0
WHEEL_DYNAMIC_FRICTION = 0.9
# wheel links have no authored mass; PhysX derives it from the colliders, so keep the original asset's value explicitly
# (the cylinder colliders would otherwise make each wheel 0.60 kg instead of 0.23 kg)
WHEEL_MASS = 0.232  # [kg]

stage = Usd.Stage.Open(SRC)
root = stage.GetDefaultPrim()
rp = root.GetPath()

# --- articulation root on the chassis
for api in (UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI):
    if root.HasAPI(api):
        root.RemoveAPI(api)
chassis = stage.GetPrimAtPath(rp.AppendPath("base_link/base"))
UsdPhysics.ArticulationRootAPI.Apply(chassis)
PhysxSchema.PhysxArticulationAPI.Apply(chassis)

# --- wheel material
mat_path = rp.AppendPath("wheel_material")
mat = UsdShade.Material.Define(stage, mat_path)
phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
phys_mat.CreateStaticFrictionAttr(WHEEL_STATIC_FRICTION)
phys_mat.CreateDynamicFrictionAttr(WHEEL_DYNAMIC_FRICTION)
phys_mat.CreateRestitutionAttr(0.0)

cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
)
for w in WHEELS:
    link = stage.GetPrimAtPath(rp.AppendPath(f"base_link/chassis_base_link/wheels/{w}_wheel_link"))
    mesh = stage.GetPrimAtPath(link.GetPath().AppendChild(f"{w}_wheel"))
    bolt = stage.GetPrimAtPath(link.GetPath().AppendChild(f"{w}_wheel_bolt"))
    rng = cache.ComputeRelativeBound(mesh, link).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    radius = 0.25 * ((hi[0] - lo[0]) + (hi[2] - lo[2]))
    width = hi[1] - lo[1]
    center = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
    # disable the old colliders (keep them for visuals)
    for prim in (mesh, bolt):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    # plain cylinder collider, invisible
    cyl = UsdGeom.Cylinder.Define(stage, link.GetPath().AppendChild("wheel_collider"))
    cyl.CreateAxisAttr(UsdGeom.Tokens.y)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(width)
    cyl.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.XformCommonAPI(cyl).SetTranslate(center)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(WHEEL_MASS)
    # velocity-controllable wheel joint
    joint = stage.GetPrimAtPath(link.GetPath().AppendChild("RevoluteJoint"))
    drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(1.0)
    print(
        f"[make_wheeled_usd] {w}: cylinder r={radius:.4f} m, width={width:.4f} m,"
        f" center={[round(c, 4) for c in center]}"
    )

stage.GetRootLayer().Export(DST)
print(f"[make_wheeled_usd] wrote {DST}")
app.close()
