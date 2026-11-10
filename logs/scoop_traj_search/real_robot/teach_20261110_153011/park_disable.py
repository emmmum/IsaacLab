# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move 101/102 from the latched pose to the park rest (Kp 20/50 + gravity FF), unload onto the rest, then disable.
Usage: park_disable.py PREV_OUT   (PREV_OUT has FINAL_CMD_DEG of the latched command)"""
import json, math, sys, time
import can
sys.path.insert(0, "/home/jetson/Downloads")
from pcan_cybergear import CANMotorController
R, D = math.radians, math.degrees
PARK = {101: R(-61.87), 102: R(-86.13)}          # rest pose measured 2026-11-10 (arm resting on its support)
KP = {101: 20.0, 102: 50.0}; KD = 3.0; ERR = R(20)
gp = json.load(open("/home/jetson/robomaster_test/gravity_params_real.json")); b1, b2 = gp["tau101_coef"], gp["tau102_coef"]
def grav(q1, q2):
    phi = q2 - q1
    return {101: b1[0]*math.sin(q1) + b1[1]*math.cos(q1) + b1[2]*math.cos(phi) + b1[3]*math.sin(phi) + b1[4],
            102: b2[0]*math.cos(phi) + b2[1]*math.sin(phi) + b2[2]}
fin = [l for l in open(sys.argv[1]).read().splitlines() if l.startswith("FINAL_CMD_DEG")][-1].split()
cmd = {101: R(float(fin[1])), 102: R(float(fin[2]))}; q = dict(cmd)
bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
M = {m: CANMotorController(bus, motor_id=m, main_can_id=254) for m in (101, 102)}
def say(s): print(time.strftime("%T ") + s, flush=True)
def send(m, tau, tgt, kp, kd):
    r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=0.0, Kp=kp, Kd=kd)
    if r[0] != m: r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=0.0, Kp=kp, Kd=kd)
    if r[0] == m: q[m] = r[1]
    return r
def tick(scale=1.0, check=True):
    g = grav(q[101], q[102])
    for m in (101, 102):
        send(m, scale * g[m], cmd[m], scale * KP[m], KD)
        if check and abs(q[m] - cmd[m]) > ERR: raise RuntimeError(f"{m} err {D(q[m]-cmd[m]):+.1f} deg")
    time.sleep(0.003)
def move(goal, T):
    s0 = dict(cmd); t0 = time.time()
    while True:
        s = min((time.time() - t0) / T, 1.0); s5 = 6*s**5 - 15*s**4 + 10*s**3
        for m in goal: cmd[m] = s0[m] + (goal[m] - s0[m]) * s5
        tick()
        if s >= 1.0: break
try:
    tick(); say(f"start q {D(q[101]):+.2f} {D(q[102]):+.2f}")
    move({102: R(-45)}, 10.0); say(f"elbow -45 -> q {D(q[101]):+.2f} {D(q[102]):+.2f}")
    move({101: PARK[101]}, 10.0); say(f"shoulder park -> q {D(q[101]):+.2f} {D(q[102]):+.2f}")
    move({102: PARK[102]}, 10.0); say(f"elbow park -> q {D(q[101]):+.2f} {D(q[102]):+.2f}")
    t0 = time.time()
    while time.time() - t0 < 1.0: tick()
    t0 = time.time()                                    # unload onto the rest: Kp and FF -> 0 over 3 s
    while (u := (time.time() - t0) / 3.0) < 1.0: tick(1.0 - u, check=False)
    for m in (101, 102): M[m].disable()
    time.sleep(0.5)
    r = {m: M[m].disable() for m in (101, 102)}
    say(f"DISABLED at q {D(r[101][1]):+.2f} {D(r[102][1]):+.2f} (resting on the park support)")
except Exception as e:
    say(f"ABORT {e!r} -> holding last command (NOT disabled)")
    g = grav(cmd[101], cmd[102])
    for m in (101, 102):
        try: send(m, g[m], cmd[m], KP[m], KD)
        except Exception: pass
finally:
    bus.shutdown()
