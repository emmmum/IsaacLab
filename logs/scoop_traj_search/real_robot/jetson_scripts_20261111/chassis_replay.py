# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a chassis forward trajectory (t,x_m,vx_m_s CSV) with drive_speed at 50 Hz: velocity FF + odometry position
feedback, y = 0, heading held. Usage: chassis_replay.py traj.csv SPEED START_AT_WALL out.csv  (stop file: ~/robomaster_test/STOP_BASE)"""
import csv, os, signal, sys, threading, time
import numpy as np
from robomaster import robot
traj, SPEED, START, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
assert 0.01 <= SPEED <= 1.0
d = np.genfromtxt(traj, delimiter=",", names=True); TT = d["t"] / SPEED; X = d["x_m"]; V = d["vx_m_s"] * SPEED
KX, KYAW, WMAX = 1.5, 2.0, 20.0
ERR_MAX = float(os.environ.get("ERR_MAX", "0.15"))
VMAX = float(os.environ.get("VMAX", "1.5")); assert 0 < VMAX <= 3.5
STOP = "/home/jetson/robomaster_test/STOP_BASE"
if os.path.exists(STOP): os.remove(STOP)
st = {}; lk = threading.Lock()
def on_pos(p):
    with lk: st["x"], st["y"] = p[0], p[1]; st["tp"] = time.time()
def on_att(a):
    with lk: st["yaw"] = a[0]
ep = robot.Robot(); ep.initialize(conn_type="rndis")
ep.chassis.sub_position(cs=1, freq=50, callback=on_pos); ep.chassis.sub_attitude(freq=50, callback=on_att)
f = open(OUT, "w", newline=""); w = csv.writer(f); w.writerow(["wall", "phase", "x_ref", "v_ff", "x", "y", "yaw", "vx_cmd", "wz_cmd"])
def say(s): print(time.strftime("%T ") + s, flush=True)
stop = [None]
for sg in (signal.SIGINT, signal.SIGTERM): signal.signal(sg, lambda *a: stop.__setitem__(0, "signal"))
try:
    t_w = time.time()
    while "x" not in st or "yaw" not in st:
        if time.time() - t_w > 5: raise RuntimeError("no odometry")
        time.sleep(0.02)
    with lk: x0, y0, yaw0 = st["x"], st["y"], st["yaw"]
    say(f"odom ok, waiting {START - time.time():.1f} s for start; duration {TT[-1]:.1f} s at x{SPEED}")
    if START - time.time() > 60 or START < time.time(): raise RuntimeError("bad start time")
    while time.time() < START:
        if stop[0] or os.path.exists(STOP): raise RuntimeError("stopped before start")
        time.sleep(0.005)
    say("playing")
    while (t := time.time() - START) <= TT[-1] + 1.0:
        if stop[0] or os.path.exists(STOP): raise RuntimeError("stop requested")
        with lk: x, y, yaw, tp = st["x"] - x0, st["y"] - y0, st["yaw"], st["tp"]
        if time.time() - tp > 0.3: raise RuntimeError("odometry stale")
        xr = float(np.interp(t, TT, X)); vf = float(np.interp(t, TT, V)) if t <= TT[-1] else 0.0
        if abs(xr - x) > ERR_MAX: raise RuntimeError(f"tracking err {xr - x:+.3f} m")
        vx = float(np.clip(vf + KX * (xr - x), -VMAX, VMAX))
        dyaw = (yaw - yaw0 + 180) % 360 - 180; wz = float(np.clip(-KYAW * dyaw, -WMAX, WMAX))
        ep.chassis.drive_speed(x=vx, y=0, z=wz, timeout=0.3)
        w.writerow([f"{time.time():.4f}", "play", f"{xr:.4f}", f"{vf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{yaw:.2f}", f"{vx:.4f}", f"{wz:.2f}"])
        time.sleep(0.02)
    say(f"DONE x {x:+.3f} (ref {xr:+.3f}) y {y:+.3f} dyaw {dyaw:+.2f}")
except Exception as e:
    say(f"ABORT {e!r}")
finally:
    for _ in range(10):
        try: ep.chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
        except Exception: pass
        time.sleep(0.02)
    try: ep.chassis.unsub_position(); ep.chassis.unsub_attitude()
    except Exception: pass
    f.close(); ep.close(); say("chassis released"); os._exit(0)
