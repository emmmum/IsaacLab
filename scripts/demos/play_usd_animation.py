# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open an animated USD in the Isaac Sim window and play its timeline in a loop.

Works with any USD whose motion is stored as time samples, e.g. the recordings written by
``scripts/demos/robomaster_lacrosse_scoop.py --save_usd``. The same file can also be opened by hand in Isaac Sim
(File > Open, then press Play).

Usage:
  ./isaaclab.sh -p scripts/demos/play_usd_animation.py --usd scoop_anim.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an animated USD in Isaac Sim.")
parser.add_argument("--usd", required=True, help="USD file with time-sampled animation")
parser.add_argument("--camera", default="/World/DemoCamera", help="camera prim to look through, if it exists")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import omni.timeline
import omni.usd

omni.usd.get_context().open_stage(os.path.abspath(args_cli.usd))
for _ in range(10):  # let the stage load and the viewport pick it up
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath(args_cli.camera).IsValid():
    from omni.kit.viewport.utility import get_active_viewport

    get_active_viewport().camera_path = args_cli.camera

timeline = omni.timeline.get_timeline_interface()
timeline.set_looping(True)
timeline.play()
while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
