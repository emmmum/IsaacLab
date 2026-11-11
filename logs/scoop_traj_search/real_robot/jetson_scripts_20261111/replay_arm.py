# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a taught arm trajectory (trimmed 100 Hz CSV) on 101/102 at a speed factor, Kp 20/50 + gravity FF + velocity FF.
Usage: replay_arm.py traj.csv SPEED out.csv PREV_OUT   (e.g. SPEED 0.1; PREV_OUT has FINAL_CMD_DEG). Ends latched at the final pose."""
import json, math, sys, time
import numpy as np, can
sys.path.insert(0, "/home/jetson/Downloads")
from pcan_cybergear import CANMotorController
traj, SPEED, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
assert 0.01 <= SPEED <= 1.0
d = np.genfromtxt(traj, delimiter=",", names=True)
T = d["t"] / SPEED; Q = {101: np.radians(d["q101_deg"]), 102: np.radians(d["q102_deg"])}
QD = {m: np.gradient(Q[m], T) for m in Q}
KP = {101: 20.0, 102: 50.0}; KD = 3.0; ERR = math.radians(20); TAU_LIM = float(__import__("os").environ.get("TAU_LIM", "4.0")); assert 0 < TAU_LIM <= 12.0
gp = json.load(open("/home/jetson/robomaster_test/gravity_params_real.json")); b1, b2 = gp["tau101_coef"], gp["tau102_coef"]
def grav(q1, q2):
    phi = q2 - q1
    return {101: b1[0]*math.sin(q1) + b1[1]*math.cos(q1) + b1[2]*math.cos(phi) + b1[3]*math.sin(phi) + b1[4],
            102: b2[0]*math.cos(phi) + b2[1]*math.sin(phi) + b2[2]}
bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
M = {m: CANMotorController(bus, motor_id=m, main_can_id=254) for m in (101, 102)}
q = {}; cmd = {}; log = open(OUT, "w"); log.write("wall_t,phase,cmd101,q101,tau101,cmd102,q102,tau102\n")
def send(m, tau, tgt, vel):
    r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=vel, Kp=KP[m], Kd=KD)
    if r[0] != m: r = M[m].send_motor_control_command(torque=tau, target_angle=tgt, target_velocity=vel, Kp=KP[m], Kd=KD)
    if r[0] != m: raise RuntimeError(f"no feedback {m}")
    q[m] = r[1]; return r
def tick(phase, vel={101: 0.0, 102: 0.0}):
    g = grav(q.get(101, cmd[101]), q.get(102, cmd[102])); rr = {}
    for m in (101, 102):
        rr[m] = send(m, g[m], cmd[m], vel[m])
        if abs(q[m] - cmd[m]) > ERR: raise RuntimeError(f"{m} tracking err {math.degrees(q[m]-cmd[m]):+.1f} deg")
    log.write(f"{time.time():.4f},{phase},{cmd[101]:.5f},{q[101]:.5f},{rr[101][3]:.3f},{cmd[102]:.5f},{q[102]:.5f},{rr[102][3]:.3f}\n")
    time.sleep(0.003)
def move_to(goal, dur, phase):
    s0 = dict(cmd); t0 = time.time()
    while True:
        s = min((time.time() - t0) / dur, 1.0); s5 = 6*s**5 - 15*s**4 + 10*s**3
        for m in goal: cmd[m] = s0[m] + (goal[m] - s0[m]) * s5
        tick(phase)
        if s >= 1.0: break
def say(s): print(time.strftime("%T ") + s, flush=True)
try:
    for m in (101, 102): M[m].write_single_param("limit_torque", TAU_LIM)
    # start from the previous controller's latched command (FINAL_CMD_DEG in its .out file) -> no jump
    fin = [l for l in open("/home/jetson/robomaster_test/" + sys.argv[4]).read().splitlines() if l.startswith("FINAL_CMD_DEG")] if len(sys.argv) > 4 else []
    if fin: cmd = {101: math.radians(float(fin[-1].split()[1])), 102: math.radians(float(fin[-1].split()[2]))}
    else: cmd = {101: Q[101][0], 102: Q[102][0]}
    tick("hold")
    say(f"start q {math.degrees(q[101]):+.2f} {math.degrees(q[102]):+.2f}; traj start {math.degrees(Q[101][0]):+.2f} {math.degrees(Q[102][0]):+.2f}, "
        f"duration {T[-1]:.1f} s at x{SPEED}")
    move_to({101: Q[101][0], 102: Q[102][0]}, 3.0, "approach")
    t1 = time.time()
    while time.time() - t1 < 1.0: tick("approach")
    import os
    if os.environ.get("START_AT"):                       # synchronised start with chassis_replay.py
        st_at = float(os.environ["START_AT"]); assert 0 <= st_at - time.time() < 60, "bad START_AT"
        while time.time() < st_at: tick("wait")
    say("playing"); t0 = time.time(); lp = 0; t_prev = 0.0
    while (t := time.time() - t0) <= T[-1]:
        if t - t_prev > 0.1:                              # loop stalled: do not jump the command, freeze and stop the base
            open("/home/jetson/robomaster_test/STOP_BASE", "w").close()
            raise RuntimeError(f"control loop stalled {t - t_prev:.3f} s at t {t:.2f}")
        t_prev = t
        for m in (101, 102): cmd[m] = float(np.interp(t, T, Q[m]))
        tick("play", {m: float(np.interp(t, T, QD[m])) for m in (101, 102)})
        if t - lp > 10: lp = t; say(f"t {t:5.1f}/{T[-1]:.1f}s cmd {math.degrees(cmd[101]):+6.2f} {math.degrees(cmd[102]):+6.2f} q {math.degrees(q[101]):+6.2f} {math.degrees(q[102]):+6.2f}")
    t1 = time.time()
    while time.time() - t1 < 1.0: tick("end")
    say(f"DONE latched at cmd {math.degrees(cmd[101]):+.2f} {math.degrees(cmd[102]):+.2f}, q {math.degrees(q[101]):+.2f} {math.degrees(q[102]):+.2f} (Kp 20/50 + FF)")
    print(f"FINAL_CMD_DEG {math.degrees(cmd[101]):.3f} {math.degrees(cmd[102]):.3f}", flush=True)
except Exception as e:
    say(f"ABORT {e!r} -> freezing at last command")
    g = grav(cmd[101], cmd[102])
    for m in (101, 102):
        try: send(m, g[m], cmd[m], 0.0)
        except Exception: pass
    print(f"FINAL_CMD_DEG {math.degrees(cmd[101]):.3f} {math.degrees(cmd[102]):.3f}", flush=True)
finally:
    log.close(); bus.shutdown()
