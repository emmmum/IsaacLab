# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a Franka Research 3 holding the lacrosse stick as one articulation (``fr3_lacrosse.usd``).

The FR3 comes from the Isaac Sim asset library (referenced, not copied). The stick geometry, materials and SDF
colliders are copied from the RoboMaster asset; the stick prim uses the RoboMaster ``lowerarm`` body frame (shaft along
+x, elbow at x = 0), so recorded stick poses apply directly. The stick is an articulation link welded to ``fr3_hand``
with a fixed joint at the grip given on the command line (TCP pose in the stick frame, as planned by
``logs/scoop_traj_search/fr3/plan_fr3.py``). Mass properties are the real stick's (0.457 kg, CoM 0.142 m ahead of the
elbow, slender-rod inertia).

Usage:
  ./isaaclab.sh -p source/isaaclab_assets/data/robomaster_lacrosse/make_fr3_lacrosse_usd.py \
      --plan logs/scoop_traj_search/fr3/fr3_plan.npz
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--plan", required=True, help="planner output with T_tcp_in_stick")
parser.add_argument("--out", default="fr3_lacrosse.usd", help="output file name, next to this script")
parser.add_argument(
    "--bend_deg",
    type=float,
    default=0.0,
    help="model error: the real stick's head is pitched by this much (rotation about the stick y axis at the throat)",
)
parser.add_argument(
    "--shift", type=float, default=0.0, help="model error: the real stick sits this far further out [m]"
)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "robomaster_lacrosse_long_stick_wheeled.usd")
DST = os.path.join(HERE, args.out)
FR3 = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd"
STICK_MASS = 0.457  # [kg]
STICK_COM = (0.142, 0.016, 0.48)  # [m] in the stick (lowerarm) frame
STICK_INERTIA = (0.0005, 0.457 * 1.146**2 / 12.0, 0.457 * 1.146**2 / 12.0)  # [kg*m^2] axial, perpendicular x2
HAND_TO_TCP = 0.1034  # [m] along the hand z axis

# --- stick geometry + materials, copied out of the (flattened) RoboMaster asset
src = Usd.Stage.Open(SRC).Flatten()
stage = Usd.Stage.CreateNew(DST)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = stage.DefinePrim("/fr3", "Xform")
root.GetReferences().AddReference(FR3)
stage.SetDefaultPrim(root)
layer = stage.GetRootLayer()
stick_path = Sdf.Path("/fr3/stick")
UsdGeom.Xform.Define(stage, stick_path)
pairs = (
    (
        "/robomaster/lowerarm/Nike_Lunar_Elite_Womens_Lacrosse_Stick",
        "/fr3/stick/Nike_Lunar_Elite_Womens_Lacrosse_Stick",
    ),
    ("/robomaster/stick_looks", "/fr3/stick/Looks"),
)
for s, d in pairs:
    Sdf.CopySpec(src, s, layer, d)
for prim in Usd.PrimRange(stage.GetPrimAtPath(stick_path)):
    for rel in prim.GetRelationships():
        targets = [Sdf.Path(str(t).replace("/robomaster/stick_looks", "/fr3/stick/Looks")) for t in rel.GetTargets()]
        if targets:
            rel.SetTargets(targets)
    # texture paths are relative to the RoboMaster asset, which sits in the same directory
stick = stage.GetPrimAtPath(stick_path)
UsdPhysics.RigidBodyAPI.Apply(stick)
mass = UsdPhysics.MassAPI.Apply(stick)
mass.CreateMassAttr(STICK_MASS)
mass.CreateCenterOfMassAttr(Gf.Vec3f(*STICK_COM))
mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*STICK_INERTIA))
mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

# --- weld the stick to the hand at the planned grip: T_hand_stick = T_hand_tcp @ inv(T_tcp_in_stick)
T_tcp_in_stick = np.load(args.plan)["T_tcp_in_stick"].astype(float)
T_hand_tcp = np.eye(4)
T_hand_tcp[2, 3] = HAND_TO_TCP
T_hand_stick = T_hand_tcp @ np.linalg.inv(T_tcp_in_stick)
# model error (the planner does not know about it): pitch the stick about the throat, where the head joins the shaft,
# and shift it along the shaft. A 10 deg bend moves the lip ~6 cm.
throat = np.array([0.28, 0.016, 0.48])
c, s_ = np.cos(np.radians(args.bend_deg)), np.sin(np.radians(args.bend_deg))
E = np.eye(4)
E[:3, :3] = [[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]
E[:3, 3] = throat - E[:3, :3] @ throat + np.array([args.shift, 0.0, 0.0])
T_hand_stick = T_hand_stick @ E
q = Gf.Matrix4d(T_hand_stick.T.tolist()).ExtractRotationQuat()  # Gf uses row vectors
joint = UsdPhysics.FixedJoint.Define(stage, "/fr3/stick_grip_joint")
joint.CreateBody0Rel().SetTargets(["/fr3/fr3_hand"])
joint.CreateBody1Rel().SetTargets([stick_path])
joint.CreateLocalPos0Attr(Gf.Vec3f(*T_hand_stick[:3, 3].tolist()))
joint.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
# the stick is welded inside the open fingers: no self-collision between the stick and the arm
art = stage.GetPrimAtPath("/fr3")
PhysxSchema.PhysxArticulationAPI.Apply(art).CreateEnabledSelfCollisionsAttr(False)
layer.Save()
print(f"[make_fr3_lacrosse_usd] wrote {DST}; stick in hand frame: pos {T_hand_stick[:3, 3].round(4)}")
app.close()
