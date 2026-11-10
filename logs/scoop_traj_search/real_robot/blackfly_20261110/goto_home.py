# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slowly return the chassis to a home pose in the power-on odometry frame (sub_position cs=1).
Usage: goto_home.py X Y HEADING_DEG   (heading = atan2 of forward displacement in that frame)"""
import math, os, sys, threading, time
from robomaster import robot
X, Y, H = float(sys.argv[1]), float(sys.argv[2]), math.radians(float(sys.argv[3]))
V, W = 0.08, 10.0                       # max m/s, deg/s
st = {"p": None, "yaw": None}; lk = threading.Lock()
ep = robot.Robot(); ep.initialize(conn_type="rndis")
ep.chassis.sub_position(cs=1, freq=50, callback=lambda d: st.__setitem__("p", (d[0], d[1])))
ep.chassis.sub_attitude(freq=50, callback=lambda d: st.__setitem__("yaw", d[0]))
def stop():
    for _ in range(10): ep.chassis.drive_speed(x=0, y=0, z=0, timeout=0.3); time.sleep(0.02)
    time.sleep(0.5)
def wait_state():
    while st["p"] is None or st["yaw"] is None: time.sleep(0.05)
def move_fwd(d):
    p0 = st["p"]; sgn = 1 if d > 0 else -1; t0 = time.time()
    while math.dist(st["p"], p0) < abs(d) - 0.004 and time.time() - t0 < 4:
        ep.chassis.drive_speed(x=sgn * 0.06, y=0, z=0, timeout=0.3); time.sleep(0.02)
    stop(); p1 = st["p"]; return (p1[0] - p0[0], p1[1] - p0[1])
def heading():
    d = move_fwd(0.08); move_fwd(-0.08); return math.atan2(d[1], d[0])
def rotate(dyaw_att, zsgn):
    a0 = st["yaw"]; t0 = time.time(); sgn = 1 if dyaw_att > 0 else -1
    while True:
        e = dyaw_att - ((st["yaw"] - a0 + 180) % 360 - 180)
        if abs(e) < 0.5 or time.time() - t0 > 8: break
        ep.chassis.drive_speed(x=0, y=0, z=zsgn * math.copysign(min(W, max(3.0, 2 * abs(e))), e), timeout=0.3); time.sleep(0.02)
    stop(); return (st["yaw"] - a0 + 180) % 360 - 180
try:
    wait_state(); time.sleep(0.5)
    print(f"start p {st['p']} yaw_att {st['yaw']:.2f}", flush=True)
    h1 = heading(); a0 = st["yaw"]
    for _ in range(25): ep.chassis.drive_speed(x=0, y=0, z=10, timeout=0.3); time.sleep(0.02)   # probe +z for 0.5 s
    stop(); da = (st["yaw"] - a0 + 180) % 360 - 180
    h2 = heading(); dh = math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
    print(f"probe: +z cmd -> att yaw {da:+.2f} deg, odom heading {dh:+.2f} deg (h {math.degrees(h1):.2f} -> {math.degrees(h2):.2f})", flush=True)
    assert abs(da) > 2 and abs(dh) > 1, "probe rotation too small"
    k = 1 if dh * da > 0 else -1          # odom heading change per att yaw change sign
    zs = 1 if da > 0 else -1              # att yaw change per +z cmd sign
    need = math.degrees((H - h2 + math.pi) % (2 * math.pi) - math.pi)
    assert abs(need) < 45, f"refusing to rotate {need:.1f} deg"
    got = rotate(k * need, zs); h3 = heading()
    print(f"rotate: need {need:+.2f} deg (heading), att moved {got:+.2f}; heading now {math.degrees(h3):.2f} (target {math.degrees(H):.2f})", flush=True)
    # translate in body frame with P control
    e0 = math.dist(st["p"], (X, Y)); t0 = time.time()
    assert e0 < 0.5, f"too far {e0:.3f} m"
    while True:
        ex, ey = X - st["p"][0], Y - st["p"][1]; e = math.hypot(ex, ey)
        if e < 0.008 or time.time() - t0 > 15: break
        if e > e0 + 0.05: print("ERROR growing, abort", flush=True); break
        bx = math.cos(h3) * ex + math.sin(h3) * ey; by = -math.sin(h3) * ex + math.cos(h3) * ey
        s = min(V, 1.5 * e) / e
        ep.chassis.drive_speed(x=bx * s, y=by * s, z=0, timeout=0.3); time.sleep(0.02)
    stop(); print(f"final p {tuple(round(v, 4) for v in st['p'])} target ({X}, {Y}) err {math.dist(st['p'], (X, Y))*1e3:.1f} mm", flush=True)
finally:
    try: stop(); ep.chassis.unsub_position(); ep.chassis.unsub_attitude()
    except Exception: pass
    ep.close(); os._exit(0)
