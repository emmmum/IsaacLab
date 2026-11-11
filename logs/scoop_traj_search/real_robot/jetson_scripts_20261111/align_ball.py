# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the chassis so the ball sits at the reference position relative to the camera (from a ball_floor json).
Usage: align_ball.py ref_ball_floor.json [tol_m]
A 20 Hz thread holds the heading (attitude yaw, SDK mode makes the chassis drift ~0.4 deg/s otherwise) and servoes the
odometry position to a target; each camera measurement (ball_locate.py + floor_fit.py) shifts the target. SDK closed at the end."""
import json, subprocess, sys, threading, time
from robomaster import robot
H = "/home/jetson/robomaster_test/"
ref = json.load(open(sys.argv[1])); TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
KP, VMAX, VMIN, KYAW, WMAX = 1.5, 0.12, 0.03, 2.0, 15.0
def pos(d): return 0.5 * (d["ball_size"]["forward"] + d["ball_depth"]["forward"]), 0.5 * (d["ball_size"]["left"] + d["ball_depth"]["left"])
rf, rl = pos(ref)
def measure():
    for _ in range(3):
        if subprocess.run(["python3", H + "ball_locate.py"], cwd=H, capture_output=True).returncode == 0 and \
           subprocess.run(["python3", H + "floor_fit.py"], cwd=H, capture_output=True).returncode == 0:
            return pos(json.load(open(H + "ball_floor.json")))
        time.sleep(0.5)
    raise RuntimeError("ball not found")
def say(s): print(time.strftime("%T ") + s, flush=True)
st = {}; tgt = {"x": None, "y": None}; run = threading.Event(); run.set(); lk = threading.Lock()
def on_pos(p):
    with lk: st["x"], st["y"] = p[0], p[1]
def on_att(a):
    with lk: st["yaw"] = a[0]
def clip(v, m): return max(-m, min(m, v))
def ctrl():
    while run.is_set():
        with lk: x, y, yaw = st["x"], st["y"], st["yaw"]
        ex, ey = tgt["x"] - x, tgt["y"] - y
        vx, vy = clip(KP * ex, VMAX), clip(KP * ey, VMAX)
        vx = 0.0 if abs(ex) < 0.003 else (vx if abs(vx) >= VMIN else VMIN * (1 if vx > 0 else -1))
        vy = 0.0 if abs(ey) < 0.003 else (vy if abs(vy) >= VMIN else VMIN * (1 if vy > 0 else -1))
        dyaw = (yaw - yaw0 + 180) % 360 - 180
        ep.chassis.drive_speed(x=vx, y=vy, z=clip(-KYAW * dyaw, WMAX), timeout=0.3); time.sleep(0.05)
ep = robot.Robot(); ep.initialize(conn_type="rndis")
ep.chassis.sub_position(cs=1, freq=20, callback=on_pos); ep.chassis.sub_attitude(freq=20, callback=on_att)
th = None
try:
    t0 = time.time()
    while not all(k in st for k in ("x", "y", "yaw")):
        if time.time() - t0 > 5: raise RuntimeError("no odometry")
        time.sleep(0.05)
    with lk: yaw0 = st["yaw"]; tgt["x"], tgt["y"] = st["x"], st["y"]
    th = threading.Thread(target=ctrl, daemon=True); th.start(); say(f"heading hold on at yaw {yaw0:.2f}")
    prev = None
    for it in range(10):
        f, l = measure(); dx, dl = f - rf, l - rl; e = max(abs(dx), abs(dl))
        with lk: dyaw = (st["yaw"] - yaw0 + 180) % 360 - 180
        say(f"iter {it}: ball fwd {f:.3f} left {l:+.3f} | err fwd {dx:+.3f} left {dl:+.3f} | dyaw {dyaw:+.2f}")
        if abs(dx) < TOL and abs(dl) < TOL: say("ALIGNED"); break
        if abs(dx) > 0.6 or abs(dl) > 0.4: raise RuntimeError("offset too large, not moving")
        if prev is not None and e > prev + 0.03: raise RuntimeError("error growing, stopping")
        prev = e
        # ball too far -> drive forward (+x); ball too far left -> drive left (odometry/SDK y is right-positive)
        tgt["x"] += dx; tgt["y"] += -dl
        t1 = time.time()
        while time.time() - t1 < 6:
            with lk: ok = abs(tgt["x"] - st["x"]) < 0.004 and abs(tgt["y"] - st["y"]) < 0.004
            if ok: break
            time.sleep(0.05)
        time.sleep(0.5)
    else: say("NOT aligned after 10 iterations")
finally:
    run.clear()
    if th: th.join(1)
    for _ in range(5): ep.chassis.drive_speed(0, 0, 0); time.sleep(0.02)
    try: ep.chassis.unsub_position(); ep.chassis.unsub_attitude()
    except Exception: pass
    ep.close(); say("SDK closed")
