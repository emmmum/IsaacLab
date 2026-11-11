# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recover from the 102 one-rotor-turn jump: hold with a software offset, move to the true 0/0, re-zero 102 there.

q102_true = q102_read - OFF (OFF = 360/7.75 deg). Leaves both motors latched at 0/0 (Kp 20/50 + gravity FF).
"""
import json, math, sys, time
import can
sys.path.insert(0, "/home/jetson/Downloads")
from pcan_cybergear import CANMotorController
R = math.radians; OFF = {101: 0.0, 102: R(360 / 7.75)}
KPH = {101: 20.0, 102: 50.0}; KD = 3.0; ERR = 0.35
gp = json.load(open("/home/jetson/robomaster_test/gravity_params_real.json")); b1, b2 = gp["tau101_coef"], gp["tau102_coef"]
def grav(q1, q2):
    phi = q2 - q1
    return {101: b1[0]*math.sin(q1) + b1[1]*math.cos(q1) + b1[2]*math.cos(phi) + b1[3]*math.sin(phi) + b1[4],
            102: b2[0]*math.cos(phi) + b2[1]*math.sin(phi) + b2[2]}
bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
M = {m: CANMotorController(bus, motor_id=m, main_can_id=254) for m in (101, 102)}
qt = {}; cmd = {}; kp = {101: 0.0, 102: 0.0}
def say(s): print(s, flush=True); open("/home/jetson/robomaster_test/enc_test.log", "a").write(time.strftime("%F %T ") + s + "\n")
def send(m, tau, tgt_true, k, kd=KD):
    r = M[m].send_motor_control_command(torque=tau, target_angle=tgt_true + OFF[m], target_velocity=0.0, Kp=k, Kd=kd)
    if r[0] != m: r = M[m].send_motor_control_command(torque=tau, target_angle=tgt_true + OFF[m], target_velocity=0.0, Kp=k, Kd=kd)
    if r[0] != m: raise RuntimeError(f"no feedback {m}")
    qt[m] = r[1] - OFF[m]; return r
def tick():
    g = grav(qt[101], qt[102])
    for m in (101, 102):
        send(m, g[m], cmd[m], kp[m])
        if kp[m] > 5 and abs(qt[m] - cmd[m]) > ERR: raise RuntimeError(f"{m} err {math.degrees(qt[m]-cmd[m]):+.1f} deg")
    time.sleep(0.002)
def move(goal, T):
    s0 = dict(cmd); t0 = time.time()
    while True:
        s = min((time.time() - t0) / T, 1.0); s5 = 6*s**5 - 15*s**4 + 10*s**3
        for m in goal: cmd[m] = s0[m] + (goal[m] - s0[m]) * s5
        tick()
        if s >= 1.0: break
try:
    for m in (101, 102):
        r = M[m].disable(); qt[m] = r[1] - OFF[m]
    say(f"[fix] read (true frame): 101 {math.degrees(qt[101]):+.2f}  102 {math.degrees(qt[102]):+.2f} (raw {math.degrees(qt[102]+OFF[102]):+.2f})")
    for m in (101, 102): M[m].write_single_param("limit_torque", 4.0)
    for m in (101, 102): M[m].enable()
    cmd = dict(qt)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        u = (time.time() - t0) / 2.0
        for m in kp: kp[m] = u * KPH[m]
        tick()
    kp = dict(KPH)
    say(f"[fix] holding at {math.degrees(qt[101]):+.2f} / {math.degrees(qt[102]):+.2f}")
    move({102: max(cmd[102], R(-45))}, 10.0); say(f"[fix] elbow up -> q {math.degrees(qt[101]):+.2f} {math.degrees(qt[102]):+.2f}")
    move({101: 0.0}, 10.0);                  say(f"[fix] shoulder 0 -> q {math.degrees(qt[101]):+.2f} {math.degrees(qt[102]):+.2f}")
    move({102: 0.0}, 10.0);                  say(f"[fix] elbow 0 -> q {math.degrees(qt[101]):+.2f} {math.degrees(qt[102]):+.2f}")
    t0 = time.time()
    while time.time() - t0 < 1.0: tick()
    # re-zero 102 at the true 0: float it on the gravity FF (Kp 0), set zero, then use the new frame (OFF = 0)
    q_before = qt[102]
    for _ in range(5):
        g = grav(qt[101], qt[102]); send(101, g[101], cmd[101], kp[101]); send(102, g[102], 0.0, 0.0, 1.0); time.sleep(0.002)
    r = M[102].set_0_pos(); OFF[102] = 0.0; time.sleep(0.01)
    g = grav(qt[101], 0.0); r = send(102, g[102], 0.0, 0.0, 1.0)
    say(f"[fix] 102 re-zeroed at true {math.degrees(q_before):+.3f} deg -> new reading {math.degrees(r[1]):+.3f} deg")
    cmd[102] = qt[102]; t0 = time.time(); k0 = 0.0
    while time.time() - t0 < 1.0:
        kp[102] = KPH[102] * (time.time() - t0); tick()
    kp = dict(KPH); move({102: 0.0}, 2.0)
    t0 = time.time()
    while time.time() - t0 < 1.0: tick()
    say(f"[fix] DONE at q {math.degrees(qt[101]):+.2f} {math.degrees(qt[102]):+.2f} (latched, Kp 20/50 + FF)")
except Exception as e:
    say(f"[fix] ABORT {e!r}")
    g = grav(qt.get(101, 0), qt.get(102, 0))
    for m in (101, 102):
        try: send(m, g[m], cmd.get(m, qt.get(m, 0)), KPH[m])
        except Exception: pass
finally:
    bus.shutdown()
