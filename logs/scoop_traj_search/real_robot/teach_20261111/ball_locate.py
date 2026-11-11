# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate the ball with the onboard D435i: color segmentation + size range + depth cross-check (camera frame)."""
import cv2, numpy as np, json, time
FX, FY, CX, CY = 919.82, 919.64, 653.36, 377.48          # color 1280x720
DFX, DFY, DCX, DCY = 639.27, 639.27, 642.76, 360.68       # depth 1280x720
R = np.array([[0.999978, 0.0066059, -0.0011447], [-0.00660344, 0.999976, 0.00213376], [0.00115877, -0.00212615, 0.999997]])
T = np.array([0.014784, -0.0000509, 0.000298])           # depth -> color: p_c = R p_d + T
D_BALL = 0.0635
H = "/home/jetson/robomaster_test/"
cap = cv2.VideoCapture(2, cv2.CAP_V4L2); cap.set(3, 1280); cap.set(4, 720)
for _ in range(30): ok, f = cap.read()
cap.release()
g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
cc = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=80, param2=25, minRadius=12, maxRadius=120)
cc = [c for c in cc[0] if c[1] > 430]                     # below the horizon (floor)
u, v, rad = map(float, cc[0])
# refine: least-squares circle on Canny edges in an annulus around the Hough circle
e = cv2.Canny(g, 40, 100); ys, xs = np.nonzero(e)
r_ = np.hypot(xs - u, ys - v); k = np.abs(r_ - rad) < 6
xs, ys = xs[k].astype(float), ys[k].astype(float)
A = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]; sol = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)[0]
u, v, rad = sol[0], sol[1], float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
x, y, w, h = int(u - rad), int(v - rad), int(2 * rad), int(2 * rad); d_px = 2 * rad
c = None
Z = FX * D_BALL / d_px
P = np.array([(u - CX) / FX * Z, (v - CY) / FY * Z, Z])   # ball centre, color optical frame (x right, y down, z fwd)
# depth cross-check
import subprocess
subprocess.run("v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat='Z16 ' --stream-mmap --stream-skip=30 --stream-count=1 --stream-to=/tmp/dep.raw", shell=True, capture_output=True)
raw = np.fromfile("/tmp/dep.raw", np.uint16)
dep = raw[:1280 * 720].reshape(720, 1280) if raw.size >= 1280 * 720 else None
print("depth raw size", raw.size)
res = dict(u=float(u), v=float(v), d_px=float(d_px), n_edge=int(len(xs)), P_size=P.tolist(), range_size=float(np.linalg.norm(P)))
if dep is not None:
    pd = R.T @ (P - T); ud, vd = DFX * pd[0] / pd[2] + DCX, DFY * pd[1] / pd[2] + DCY
    r_d = DFX * D_BALL / 2 / pd[2]
    yy, xx = np.mgrid[0:720, 0:1280]; m = (xx - ud) ** 2 + (yy - vd) ** 2 < (0.5 * r_d) ** 2
    z = dep[m].astype(float) * 1e-3; z = z[z > 0]
    if len(z):
        zs = float(np.median(z)); zc = zs + D_BALL / 2 * 0.8   # front-surface patch -> centre (approx)
        Pd = np.array([(ud - DCX) / DFX * zc, (vd - DCY) / DFY * zc, zc]); Pc = R @ Pd + T
        res.update(depth_valid=int(len(z)), depth_surface=zs, P_depth=Pc.tolist(), uv_depth=[ud, vd])
    cv2.imwrite(H + "ball_depth.png", cv2.applyColorMap(cv2.convertScaleAbs(dep, alpha=255 / 3000), cv2.COLORMAP_JET))
vis = f.copy(); cv2.circle(vis, (int(u), int(v)), int(rad), (0, 0, 255), 2); cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imwrite(H + "ball_locate.jpg", vis)
res["t"] = time.strftime("%F %T"); print(json.dumps(res, indent=1))
json.dump(res, open(H + "ball_locate.json", "w"), indent=1)
