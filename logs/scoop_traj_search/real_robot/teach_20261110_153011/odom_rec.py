# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Passive chassis odometry logger (no motion commands). Usage: odom_rec.py out.csv ; stop with SIGTERM/SIGINT."""
import csv, os, signal, sys, threading, time
from robomaster import robot
ep = robot.Robot(); ep.initialize(conn_type="rndis")
f = open(sys.argv[1], "w", newline=""); w = csv.writer(f); lk = threading.Lock()
def log(*r):
    with lk: w.writerow([f"{time.time():.4f}", *r])
log("kind", "a", "b", "c")
ep.chassis.sub_position(cs=1, freq=50, callback=lambda d: log("pos", *d[:3]))
ep.chassis.sub_attitude(freq=50, callback=lambda d: log("att", *d[:3]))
ep.chassis.sub_velocity(freq=50, callback=lambda d: log("vel", *d[:3]))
stop = threading.Event()
for sg in (signal.SIGINT, signal.SIGTERM): signal.signal(sg, lambda *a: stop.set())
print("odom_rec running", flush=True); stop.wait(1800)
try: ep.chassis.unsub_position(); ep.chassis.unsub_attitude(); ep.chassis.unsub_velocity()
except Exception: pass
with lk: f.close()
ep.close(); print("odom_rec stopped", flush=True); os._exit(0)
