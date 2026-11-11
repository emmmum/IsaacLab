# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ball position from one onboard D435i snapshot (desktop port of the Jetson ball_locate.py + floor_fit.py).

Inputs: color BGR 1280x720 and depth Z16 1280x720 [mm]. Output dict matches ball_floor.json:
cam_height, pitch_down_deg, roll_deg, ball_size / ball_depth = {forward, left, height} [m] on the floor plane.
"""
import cv2
import numpy as np

FX, FY, CX, CY = 919.82, 919.64, 653.36, 377.48           # color 1280x720
DFX, DFY, DCX, DCY = 639.27, 639.27, 642.76, 360.68        # depth 1280x720
R = np.array([[0.999978, 0.0066059, -0.0011447], [-0.00660344, 0.999976, 0.00213376], [0.00115877, -0.00212615, 0.999997]])
T = np.array([0.014784, -0.0000509, 0.000298])            # depth -> color: p_c = R p_d + T
D_BALL = 0.0635


def locate(color, dep):
    g = cv2.GaussianBlur(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    cc = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=80, param2=25, minRadius=12, maxRadius=120)
    cc = [] if cc is None else [c for c in cc[0] if c[1] > 430]
    if not cc:
        raise RuntimeError("ball not found in the RealSense photo")
    u, v, rad = map(float, cc[0])
    e = cv2.Canny(g, 40, 100); ys, xs = np.nonzero(e)
    k = np.abs(np.hypot(xs - u, ys - v) - rad) < 6
    xs, ys = xs[k].astype(float), ys[k].astype(float)
    sol = np.linalg.lstsq(np.c_[2 * xs, 2 * ys, np.ones_like(xs)], xs ** 2 + ys ** 2, rcond=None)[0]
    u, v, rad = sol[0], sol[1], float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
    Z = FX * D_BALL / (2 * rad)
    P = np.array([(u - CX) / FX * Z, (v - CY) / FY * Z, Z])
    res = dict(u=float(u), v=float(v), d_px=2 * rad, P_size=P.tolist())
    pd = R.T @ (P - T); ud, vd = DFX * pd[0] / pd[2] + DCX, DFY * pd[1] / pd[2] + DCY
    yy, xx = np.mgrid[0:720, 0:1280]
    m = (xx - ud) ** 2 + (yy - vd) ** 2 < (0.5 * DFX * D_BALL / 2 / pd[2]) ** 2
    z = dep[m].astype(float) * 1e-3; z = z[z > 0]
    if len(z):
        zc = float(np.median(z)) + D_BALL / 2 * 0.8
        Pd = np.array([(ud - DCX) / DFX * zc, (vd - DCY) / DFY * zc, zc])
        res.update(P_depth=(R @ Pd + T).tolist(), uv_depth=[ud, vd])
    vis = color.copy(); cv2.circle(vis, (int(u), int(v)), int(rad), (0, 0, 255), 2)
    return res, vis


def floor(dep, res):
    d = dep.astype(float) * 1e-3
    yy, xx = np.mgrid[0:720, 0:1280]
    m = (d > 0.3) & (d < 3.0) & (yy > 450)
    if "uv_depth" in res:
        m &= np.hypot(xx - res["uv_depth"][0], yy - res["uv_depth"][1]) > 70
    P = np.c_[(xx[m] - DCX) / DFX * d[m], (yy[m] - DCY) / DFX * d[m], d[m]] @ R.T + T
    rng = np.random.default_rng(0); best = None
    for _ in range(300):
        s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0]); n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < 0.01
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]; c = Q.mean(0); n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    if n[1] > 0:
        n = -n
    h = float(-(c @ n))
    fwd = np.array([0, 0, 1.0]) - n * n[2]; fwd /= np.linalg.norm(fwd); left = np.cross(n, fwd)
    on_floor = lambda p: dict(forward=float(p @ fwd), left=float(p @ left), height=float(p @ n + h))
    out = dict(cam_height=h, pitch_down_deg=float(np.degrees(np.arcsin(-n[2]))), roll_deg=float(np.degrees(np.arctan2(n[0], -n[1]))),
               inliers=int(best.sum()), rms=float(np.std((Q - c) @ n)), ball_size=on_floor(np.array(res["P_size"])))
    if "P_depth" in res:
        out["ball_depth"] = on_floor(np.array(res["P_depth"]))
    return out


def analyse(color, dep):
    res, vis = locate(color, dep)
    out = floor(dep, res)
    out["detect"] = res
    return out, vis
