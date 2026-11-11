# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Jetson: move the chassis by (dx forward, dy LEFT) [m] and rotate by dyaw [deg, attitude-yaw units], closed loop on
odometry position + attitude yaw at 20 Hz, then stop and close the SDK. Usage: move_rel.py DX DY DYAW"""
import sys, threading, time
from robomaster import robot
dx, dy, dyaw = map(float, sys.argv[1:4])
assert abs(dx) < 0.5 and abs(dy) < 0.3 and abs(dyaw) <= 100
KP, VMAX, VMIN, KYAW, WMAX, WMIN = 1.5, 0.12, 0.03, 2.0, 15.0, 2.0
st = {}; lk = threading.Lock()
def on_pos(p):
    with lk: st["x"], st["y"] = p[0], p[1]
def on_att(a):
    with lk: st["yaw"] = a[0]
clip = lambda v, m: max(-m, min(m, v))
ep = robot.Robot(); ep.initialize(conn_type="rndis")
ep.chassis.sub_position(cs=1, freq=20, callback=on_pos); ep.chassis.sub_attitude(freq=20, callback=on_att)
try:
    t0 = time.time()
    while not all(k in st for k in ("x", "y", "yaw")):
        if time.time() - t0 > 5: raise RuntimeError("no odometry")
        time.sleep(0.05)
    with lk: tx, ty, tyaw = st["x"] + dx, st["y"] - dy, st["yaw"] + dyaw     # odometry y is right-positive
    t0 = time.time()
    while time.time() - t0 < 8 + abs(dyaw) / 12:
        with lk: ex, ey, eyaw = tx - st["x"], ty - st["y"], (tyaw - st["yaw"] + 180) % 360 - 180
        if abs(ex) < 0.003 and abs(ey) < 0.003 and abs(eyaw) < 0.2: break
        vx = 0.0 if abs(ex) < 0.003 else clip(KP * ex, VMAX); vx = vx if vx == 0 or abs(vx) >= VMIN else VMIN * (1 if vx > 0 else -1)
        vy = 0.0 if abs(ey) < 0.003 else clip(KP * ey, VMAX); vy = vy if vy == 0 or abs(vy) >= VMIN else VMIN * (1 if vy > 0 else -1)
        wz = 0.0 if abs(eyaw) < 0.2 else clip(KYAW * eyaw, WMAX); wz = wz if wz == 0 or abs(wz) >= WMIN else WMIN * (1 if wz > 0 else -1)
        ep.chassis.drive_speed(x=vx, y=vy, z=wz, timeout=0.3); time.sleep(0.05)
    with lk: print(f"done err x {tx - st['x']:+.4f} y {ty - st['y']:+.4f} yaw {(tyaw - st['yaw'] + 180) % 360 - 180:+.2f}", flush=True)
finally:
    for _ in range(5): ep.chassis.drive_speed(0, 0, 0); time.sleep(0.02)
    try: ep.chassis.unsub_position(); ep.chassis.unsub_attitude()
    except Exception: pass
    ep.close()
