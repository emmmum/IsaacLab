# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect SDK and hold zero chassis speed for DUR s; log chassis yaw. Usage: zero_hold.py DUR MODE(none|free|zero_cmd)"""
import sys, time
from robomaster import robot
dur, mode = float(sys.argv[1]), sys.argv[2]
ep = robot.Robot(); ep.initialize(conn_type="rndis"); att = {}
ep.chassis.sub_attitude(freq=10, callback=lambda a: att.update(yaw=a[0]))
try:
    if mode == "free": print("FREE", ep.set_robot_mode(mode=robot.FREE), flush=True)
    time.sleep(0.5); y0 = att.get("yaw"); t0 = time.time(); print(f"{time.time():.3f} start yaw {y0}", flush=True)
    while time.time() - t0 < dur:
        if mode in ("free", "zero_cmd"): ep.chassis.drive_speed(0, 0, 0, timeout=0.3)
        time.sleep(0.05)
    print(f"{time.time():.3f} end yaw {att.get('yaw')} -> drift {att.get('yaw') - y0:+.2f} deg in {dur:.0f} s", flush=True)
finally:
    ep.chassis.unsub_attitude(); ep.close()
