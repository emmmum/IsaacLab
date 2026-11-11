# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the car back to a session's START pose using the Blackfly chassis tag (first-frame alignment).
Measures the tag pose (1 s of track.py), expresses it in the session's start-pose frame, sends move_rel.py to the Jetson,
iterates. Attitude yaw is clockwise-positive, the Blackfly yaw counter-clockwise-positive. Usage: bf_align.py SESSION_DIR [tol_m tol_deg]"""
import os, subprocess, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); BF = os.path.join(HERE, "..", "blackfly_20261110"); PY = sys.executable
D = sys.argv[1]; TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003; TOLY = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3


def load(f, n=None):
    a = np.genfromtxt(f, delimiter=",", names=True, invalid_raise=False); a = a[a["detected"] == 1]
    return a[:n] if n else a


ref = load(f"{D}/blackfly.csv", 100)
R0 = cv2.Rodrigues(np.array([ref["rx"][0], ref["ry"][0], ref["rz"][0]]))[0]
p0 = np.array([ref["x"].mean(), ref["y"].mean(), ref["z"].mean()])


def measure():
    out = os.path.join(HERE, "sessions", ".bf_align.csv")
    subprocess.run([PY, "track.py", "1.5", "2000", out, "0"], cwd=BF, capture_output=True, timeout=30)
    a = load(out)[-150:]
    if len(a) < 50: raise RuntimeError("tag not visible")
    fw, lf, yw = [], [], []
    for r in a:
        Rr = R0.T @ cv2.Rodrigues(np.array([r["rx"], r["ry"], r["rz"]]))[0]; dp = R0.T @ (np.array([r["x"], r["y"], r["z"]]) - p0)
        fw.append(-dp[0]); lf.append(dp[2]); yw.append(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])))   # left = +tag z, yaw CCW+ (verified 13:1x)
    return float(np.median(fw)), float(np.median(lf)), float(np.median(yw))


def jetson_move(dx, dy, dyaw):
    r = subprocess.run([os.path.join(HERE, "js.sh"), f"cd ~/robomaster_test && timeout 30 python3 move_rel.py {dx:.4f} {dy:.4f} {dyaw:.3f} 2>&1 | grep done"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    return r.stdout.strip()


sign = -1.0; prev = None   # attitude yaw is clockwise-positive, Blackfly yaw (above) counter-clockwise-positive
for it in range(10):
    f, l, y = measure()
    e = max(abs(f), abs(l))
    if abs(y) > 70: print(f'yaw error {y:+.1f} deg too large, stopping', flush=True); break
    if prev is not None and e > 1.5 * prev + 0.01: print(f"error growing ({prev*1e3:.1f} -> {e*1e3:.1f} mm), stopping", flush=True); break
    prev = e
    print(f"iter {it}: car vs start pose  fwd {f * 1e3:+.1f} mm  left {l * 1e3:+.1f} mm  yaw {y:+.2f} deg", flush=True)
    if abs(f) < TOL and abs(l) < TOL and abs(y) < TOLY: print("ALIGNED", flush=True); break
    if sign is None and abs(y) >= TOLY:                       # first: rotate only, to learn the attitude-yaw sign
        print("  ", jetson_move(0, 0, -y), flush=True); time.sleep(0.5)
        y2 = measure()[2]; sign = 1.0 if abs(y2) < abs(y) else -1.0
        print(f"   yaw {y:+.2f} -> {y2:+.2f} deg => attitude sign {sign:+.0f}", flush=True)
        if sign < 0: print("  ", jetson_move(0, 0, y2), flush=True)
        continue
    print("  ", jetson_move(-f, -l, -y * (sign or 1.0) if abs(y) >= TOLY else 0.0), flush=True); time.sleep(0.5)
else:
    print("NOT aligned after 10 iterations", flush=True)
