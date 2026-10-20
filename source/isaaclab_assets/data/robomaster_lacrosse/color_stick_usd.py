# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Give the lacrosse stick of the RoboMaster-lacrosse USDs its original colors.

The stick meshes in the robot USDs come from the "Nike Lunar Elite Women's Lacrosse Stick" OBJ model (same part names,
same 1.146 m length) but were imported without materials, so they render grey. This script reads the model's ``.mtl``
file and binds one ``UsdPreviewSurface`` material per part (diffuse color or diffuse texture, roughness from the
specular exponent). Only visual materials are added: collision shapes, masses and joints are untouched, and the bound
materials carry no physics material API, so PhysX keeps using the default contact material for the stick.

Diffuse textures are copied next to the USDs into ``stick_textures/``. Normal maps are not used (the rope normal map
alone is 36 MB).

Usage:
  ./isaaclab.sh -p source/isaaclab_assets/data/robomaster_lacrosse/color_stick_usd.py --model_dir <extracted rar dir>
"""

import argparse
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model_dir", required=True, help="directory holding the extracted .obj/.mtl and textures")
args = parser.parse_args()
app = AppLauncher(headless=True).app

from pxr import Gf, Sdf, Usd, UsdShade  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
USDS = ("robomaster_lacrosse_long_stick.usd", "robomaster_lacrosse_long_stick_wheeled.usd")
TEX_DIR = "stick_textures"
STICK = "lowerarm/Nike_Lunar_Elite_Womens_Lacrosse_Stick"
# OBJ group -> material, from the ``g`` / ``usemtl`` order in the OBJ (groups without ``usemtl`` inherit the previous)
PART_MATERIAL = {
    "plastic_logo": "gray_plastic_emblem_nike",
    "bulging": "plastic_decor_green",
    "rope_1": "black_ropes",
    "rope_2": "black_ropes",
    "rope_3": "black_ropes",
    "rope_4": "black_ropes",
    "rope_5": "black_ropes",
    "rim": "black_plastic_10",
    "rim_inset": "Plastic_white_inse_t",
    "Cap": "black_plastic_bottom_cap",
    "cylindrical_insert": "Round_logo_Nike",  # "cylindrical_insert_pivot" in the OBJ
    "insert_line": "gray_plastic_strip",
    "handle": "Plastic_black_stick_Nike",
}


def read_mtl(path: str) -> dict[str, dict]:
    """Parse the diffuse color, diffuse texture and specular exponent of each material in a ``.mtl`` file."""
    mats, cur = {}, None
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "newmtl":
            cur = mats.setdefault(tok[1], {"Kd": (0.5, 0.5, 0.5), "map_Kd": None, "Ns": 32.0})
        elif cur is not None and tok[0] == "Kd":
            cur["Kd"] = tuple(float(x) for x in tok[1:4])
        elif cur is not None and tok[0] == "map_Kd" and len(tok) > 1:
            cur["map_Kd"] = tok[1]
        elif cur is not None and tok[0] == "Ns":
            cur["Ns"] = float(tok[1])
    return mats


def make_material(stage: Usd.Stage, path: Sdf.Path, spec: dict, tex_rel: str | None) -> UsdShade.Material:
    """Author a ``UsdPreviewSurface`` material, textured when ``tex_rel`` is given."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    # Blinn-Phong exponent -> roughness (Ns 16 -> 0.71, 32 -> 0.58, 64 -> 0.45)
    roughness = max(0.2, min(0.9, (2.0 / (spec["Ns"] + 2.0)) ** 0.25))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if tex_rel:
        reader = UsdShade.Shader.Define(stage, path.AppendChild("st_reader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path.AppendChild("diffuse_texture"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["Kd"]))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


mtl = [f for f in os.listdir(args.model_dir) if f.endswith(".mtl")]
assert len(mtl) == 1, f"expected one .mtl file in {args.model_dir}, found {mtl}"
specs = read_mtl(os.path.join(args.model_dir, mtl[0]))

os.makedirs(os.path.join(HERE, TEX_DIR), exist_ok=True)
for name in set(PART_MATERIAL.values()):
    tex = specs[name]["map_Kd"]
    if tex:
        shutil.copyfile(os.path.join(args.model_dir, tex), os.path.join(HERE, TEX_DIR, tex))

for usd in USDS:
    stage = Usd.Stage.Open(os.path.join(HERE, usd))
    root = stage.GetDefaultPrim().GetPath()
    looks = root.AppendPath("stick_looks")
    stage.RemovePrim(looks)  # re-running the script replaces the materials
    mats = {}
    for name in sorted(set(PART_MATERIAL.values())):
        tex = specs[name]["map_Kd"]
        mats[name] = make_material(stage, looks.AppendChild(name), specs[name], f"./{TEX_DIR}/{tex}" if tex else None)
    n = 0
    for part, name in PART_MATERIAL.items():
        part_prim = stage.GetPrimAtPath(root.AppendPath(f"{STICK}/{part}"))
        assert part_prim.IsValid(), f"{usd}: missing stick part {part}"
        for prim in Usd.PrimRange(part_prim):
            if prim.GetTypeName() == "Mesh":
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[name])
                n += 1
    stage.GetRootLayer().Save()
    print(f"[color_stick_usd] {usd}: bound {len(mats)} materials to {n} stick meshes")

app.close()
