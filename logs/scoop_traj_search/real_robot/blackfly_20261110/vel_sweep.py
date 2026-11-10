# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base velocity sweep: steps and trapezoids at several speeds (drive_speed 50 Hz), each fwd then back.
Usage: vel_sweep.py out.csv"""
import csv, os, sys, threading, time
from robomaster import robot
out = sys.argv[1]
STEPS = [(0.2, 1.5), (0.4, 1.2), (0.6, 0.9), (0.9, 0.6)]   # (speed m/s, duration s) -> <= ~0.55 m each way
TRAPS = [0.2, 0.4, 0.6, 0.9]; A, D = 1.5, 0.45              # accel m/s^2, target distance m
ep = robot.Robot(); ep.initialize(conn_type="rndis")
f = open(out, "w", newline=""); w = csv.writer(f); lock = threading.Lock()
def log(*row):
    with lock: w.writerow([f"{time.time():.4f}", *row])
log("kind", "a", "b", "c")
ep.chassis.sub_velocity(freq=50, callback=lambda d: log("vel", *d[:3]))
ep.chassis.sub_position(cs=1, freq=50, callback=lambda d: log("pos", *d[:3]))
def run(vfun, T, label):
    log("label", label); t0 = time.time()
    while (t := time.time() - t0) < T:
        v = vfun(t); assert abs(v) <= 1.0
        ep.chassis.drive_speed(x=v, y=0, z=0, timeout=0.3); log("cmd", f"{v:.4f}"); time.sleep(0.02)
    for _ in range(10): ep.chassis.drive_speed(x=0, y=0, z=0, timeout=0.3); log("cmd", "0.0"); time.sleep(0.02)
def trap(V, sgn):
    ta = V / A; tc = max((D - V * ta) / V, 0.0)
    return (lambda t: sgn * (A * t if t < ta else (V if t < ta + tc else max(V - A * (t - ta - tc), 0.0)))), 2 * ta + tc
try:
    time.sleep(1.5)
    for V, T in STEPS:
        for sgn in (1, -1):
            run(lambda t, v=sgn * V: v, T, f"step_{V}_{'fwd' if sgn > 0 else 'back'}"); time.sleep(2.0)
    for V in TRAPS:
        for sgn in (1, -1):
            fn, T = trap(V, sgn); run(fn, T, f"trap_{V}_{'fwd' if sgn > 0 else 'back'}"); time.sleep(2.0)
finally:
    try: ep.chassis.drive_speed(x=0, y=0, z=0); ep.chassis.unsub_velocity(); ep.chassis.unsub_position()
    except Exception: pass
    with lock: f.close()
    ep.close(); print("sweep done", flush=True); os._exit(0)
