# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gravity-compensation (zero-Kp) mode for 101/102 with the identified gravity model and soft joint walls.

Enter: Kp ramps from --kp-from to 0 and the gravity FF ramps in over RAMP s.
Stop : touch ~/robomaster_test/STOP_HOLD -> Kp ramps back to --kp-from holding the current q (FF kept), then exit;
       the motors latch that last command (no disable).
"""
import argparse, json, math, os, signal, sys, time
import can
sys.path.insert(0, "/home/jetson/Downloads")
from pcan_cybergear import CANMotorController

p = argparse.ArgumentParser()
p.add_argument("--init", type=float, nargs=2, required=True, help="current holder cmd [deg]")
p.add_argument("--kp-from", type=float, nargs=2, default=[20.0, 50.0])
p.add_argument("--kd", type=float, default=0.5)
p.add_argument("--params", default="/home/jetson/robomaster_test/gravity_params_real.json")
p.add_argument("--no-wall", action="store_true", help="disable the soft joint walls (no collision protection)")
a = p.parse_args()
HOME = "/home/jetson/robomaster_test"; STOP = f"{HOME}/STOP_HOLD"; LOG = f"{HOME}/grav_comp.log"
RAMP, KD_HOLD = 3.0, 3.0
WALL_KP = {101: 20.0, 102: 50.0}
# 102 safe range [deg] per 101 (sim collision sweep, ground >= 5 cm, 5 deg chassis margin)
SAFE = [(-45, -59, 0), (-40, -57, 8), (-35, -56, 15), (-30, -55, 23), (-25, -55, 30), (-20, -54, 37), (-15, -56, 43),
        (-10, -52, 50), (-5, -47, 55), (0, -42, 53), (5, -37, 54), (10, -32, 55), (15, -26, 56), (20, -20, 57),
        (25, -14, 58), (30, -7, 58), (35, -1, 57), (40, 6, 57), (45, 13, 60)]
SH_LIM = 45.0
gp = json.load(open(a.params)); b1, b2 = gp["tau101_coef"], gp["tau102_coef"]
def grav(q1, q2):
    phi = q2 - q1
    t1 = b1[0] * math.sin(q1) + b1[1] * math.cos(q1) + b1[2] * math.cos(phi) + b1[3] * math.sin(phi) + b1[4]
    t2 = b2[0] * math.cos(phi) + b2[1] * math.sin(phi) + b2[2]
    return {101: t1, 102: t2}
def el_range(sh_deg):
    sh = max(-45, min(45, sh_deg))
    for (s0, l0, h0), (s1, l1, h1) in zip(SAFE, SAFE[1:]):
        if s0 <= sh <= s1:
            u = (sh - s0) / (s1 - s0); return l0 + u * (l1 - l0), h0 + u * (h1 - h0)
def wall(m, q):   # -> (kp, target) ; kp 0 inside the safe set
    if a.no_wall: return 0.0, q[m]
    d1, d2 = math.degrees(q[101]), math.degrees(q[102])
    if m == 101:
        if d1 > SH_LIM: return WALL_KP[m], math.radians(SH_LIM)
        if d1 < -SH_LIM: return WALL_KP[m], math.radians(-SH_LIM)
        return 0.0, q[101]
    lo, hi = el_range(d1)
    if d2 > hi: return WALL_KP[m], math.radians(hi)
    if d2 < lo: return WALL_KP[m], math.radians(lo)
    return 0.0, q[102]

stop = [None]
signal.signal(signal.SIGINT, lambda *x: stop.__setitem__(0, "SIGINT"))
signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__(0, "SIGTERM"))
if os.path.exists(STOP): os.remove(STOP)
bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
M = {101: CANMotorController(bus, motor_id=101, main_can_id=254), 102: CANMotorController(bus, motor_id=102, main_can_id=254)}
KP0 = {101: a.kp_from[0], 102: a.kp_from[1]}
KP_EXIT = {101: 20.0, 102: 50.0}
cmd0 = {101: math.radians(a.init[0]), 102: math.radians(a.init[1])}
q = dict(cmd0); last = {}
log = open(LOG, "a", buffering=1)
def say(s): print(s, flush=True); log.write(f"{time.time():.3f} {s}\n")
def send(m, tau, tgt, kp, kd):
    r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=0.0, Kp=kp, Kd=kd)
    if r[0] != m:
        r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=0.0, Kp=kp, Kd=kd)
    if r[0] == m: q[m] = r[1]; last[m] = r
    return r
t0 = time.time(); lp = 0; n = 0
try:
    # read q first with the holder's command (no change)
    free_start = all(v == 0 for v in KP0.values())   # taking over from a latched Kp-0 + FF command
    g0 = grav(cmd0[101], cmd0[102])
    for m in M: send(m, g0[m], cmd0[m], KP0[m], a.kd if free_start else KD_HOLD)
    say(f"[grav] walls {'OFF' if a.no_wall else 'on'} | start q101 {math.degrees(q[101]):+.2f} q102 {math.degrees(q[102]):+.2f}")
    while stop[0] is None:
        if os.path.exists(STOP): stop[0] = "STOP_HOLD"; break
        t = time.time() - t0; u = 1.0 if free_start else min(t / RAMP, 1.0)
        g = grav(q[101], q[102])
        for m in M:
            kw, tw = wall(m, q)
            if u < 1.0:   # entering: blend holder (cmd0, KP0, KD_HOLD) -> free (Kp 0, kd)
                kp = (1 - u) * KP0[m] + u * kw; tgt = cmd0[m] if kw == 0 else tw
                kd = (1 - u) * KD_HOLD + u * a.kd; tau = g[m]   # previous holder already applies the gravity FF
            else:
                kp, tgt, kd, tau = kw, tw, a.kd, g[m]
            send(m, tau, tgt, kp, kd)
        n += 1
        if t - lp > 0.5:
            lp = t
            say(f"t={t:6.1f}s hz={n/max(t,1e-6):4.0f} | 101 q={math.degrees(q[101]):+6.2f} tau={last[101][3]:+.2f} ff={g[101]:+.2f} | "
                f"102 q={math.degrees(q[102]):+6.2f} tau={last[102][3]:+.2f} ff={g[102]:+.2f} | wall {[wall(m, q)[0] > 0 for m in M]}")
        time.sleep(0.002)
finally:
    # re-stiffen at the current position, keep the gravity FF, then leave it latched
    hold = dict(q); ts = time.time()
    while True:
        u = min((time.time() - ts) / RAMP, 1.0); g = grav(q[101], q[102])
        for m in M: send(m, g[m], hold[m], u * KP_EXIT[m], a.kd + u * (KD_HOLD - a.kd))
        if u >= 1.0: break
        time.sleep(0.002)
    say(f"[grav] stopped ({stop[0]}); holding q101 {math.degrees(hold[101]):+.2f} q102 {math.degrees(hold[102]):+.2f} "
        f"with Kp {KP_EXIT[101]}/{KP_EXIT[102]} + gravity FF (latched)")
    say(f"FINAL_CMD_DEG {math.degrees(hold[101]):.3f} {math.degrees(hold[102]):.3f}")
    bus.shutdown()
