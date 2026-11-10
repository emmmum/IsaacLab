# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base velocity step test (drive_speed 50 Hz) with chassis logging. Usage: step_test.py out.csv V T"""
import csv, os, sys, time
from robomaster import robot
out, V, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
assert abs(V) <= 0.6 and T <= 2.0
ep = robot.Robot(); ep.initialize(conn_type="rndis")
f = open(out, "w", newline=""); w = csv.writer(f); w.writerow(["wall", "kind", "a", "b", "c", "d", "e", "f"])
ep.chassis.sub_velocity(freq=50, callback=lambda d: w.writerow([f"{time.time():.4f}", "vel"] + list(d)))
ep.chassis.sub_position(cs=1, freq=50, callback=lambda d: w.writerow([f"{time.time():.4f}", "pos"] + list(d)))
def run(v, T, label):
    w.writerow([f"{time.time():.4f}", "label", label]); t0 = time.time()
    while time.time() - t0 < T:
        ep.chassis.drive_speed(x=v, y=0, z=0, timeout=0.3); w.writerow([f"{time.time():.4f}", "cmd", v]); time.sleep(0.02)
    for _ in range(10): ep.chassis.drive_speed(x=0, y=0, z=0, timeout=0.3); w.writerow([f"{time.time():.4f}", "cmd", 0.0]); time.sleep(0.02)
try:
    time.sleep(1.5); run(V, T, "step_fwd"); time.sleep(2.0); run(-V, T, "step_back"); time.sleep(1.5)
finally:
    try: ep.chassis.drive_speed(x=0, y=0, z=0); ep.chassis.unsub_velocity(); ep.chassis.unsub_position()
    except Exception: pass
    f.close(); ep.close(); print("step test done", flush=True); os._exit(0)
