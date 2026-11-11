# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np, json
DFX, DCX, DCY = 639.27, 642.76, 360.68
R = np.array([[0.999978, 0.0066059, -0.0011447], [-0.00660344, 0.999976, 0.00213376], [0.00115877, -0.00212615, 0.999997]])
T = np.array([0.014784, -0.0000509, 0.000298])
d = np.fromfile("/tmp/dep.raw", np.uint16)[:1280 * 720].reshape(720, 1280).astype(float) * 1e-3
yy, xx = np.mgrid[0:720, 0:1280]
m = (d > 0.3) & (d < 3.0) & (yy > 450)
res = json.load(open("/home/jetson/robomaster_test/ball_locate.json"))
ud, vd = res["uv_depth"]; m &= np.hypot(xx - ud, yy - vd) > 70          # exclude ball + shadow area
P = np.c_[(xx[m] - DCX) / DFX * d[m], (yy[m] - DCY) / DFX * d[m], d[m]] @ R.T + T   # -> color frame
rng = np.random.default_rng(0); best = None
for _ in range(300):
    s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0]); n /= np.linalg.norm(n)
    inl = np.abs((P - s[0]) @ n) < 0.01
    if best is None or inl.sum() > best.sum(): best = inl
Q = P[best]; c = Q.mean(0); n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
if n[1] > 0: n = -n                                   # normal pointing up (camera y is down)
h = float(-(c @ n))                                   # camera height above floor
pitch = float(np.degrees(np.arcsin(-n[2])))           # + = camera tilted down
roll = float(np.degrees(np.arctan2(n[0], -n[1])))
B = np.array(res["P_size"]); Bd = np.array(res["P_depth"])
def floor_xy(p):                                      # forward along floor, left, height of the point
    up = n; fwd = np.array([0, 0, 1.0]) - up * up[2]; fwd /= np.linalg.norm(fwd); left = np.cross(up, fwd)
    return dict(forward=float(p @ fwd), left=float(p @ left), height=float(p @ up + h))
out = dict(cam_height=h, pitch_down_deg=pitch, roll_deg=roll, inliers=int(best.sum()), rms=float(np.std((Q - c) @ n)),
           ball_size=floor_xy(B), ball_depth=floor_xy(Bd))
print(json.dumps(out, indent=1)); json.dump(out, open("/home/jetson/robomaster_test/ball_floor.json", "w"), indent=1)
