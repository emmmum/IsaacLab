# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay all replays of one session (base from the Blackfly) against the taught reference. Usage: compare_replays.py SESSION_DIR"""
import glob, os, sys
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]


def real(RD):
    TS = os.path.basename(glob.glob(f"{RD}/replay_base_*.csv")[0])[len("replay_base_"):-4]
    B = np.genfromtxt(f"{RD}/replay_base_{TS}.csv", delimiter=",", names=True, dtype=None, encoding=None); T0 = B["wall"][0]
    a = np.genfromtxt(f"{RD}/blackfly.csv", delimiter=",", names=True, invalid_raise=False); a = a[a["detected"] == 1]
    tr = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"]) + 0.047 - T0
    R0 = cv2.Rodrigues(np.array([a["rx"][0], a["ry"][0], a["rz"][0]]))[0]; p0 = np.array([a["x"][:100].mean(), a["y"][:100].mean(), a["z"][:100].mean()])
    fw, lf, yw = [], [], []
    for r in a:
        Rr = R0.T @ cv2.Rodrigues(np.array([r["rx"], r["ry"], r["rz"]]))[0]; dp = R0.T @ (np.array([r["x"], r["y"], r["z"]]) - p0)
        fw.append(-dp[0]); lf.append(dp[2]); yw.append(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])))   # left = +tag z, yaw CCW+
    return TS, tr, np.array(fw), np.array(lf), np.array(yw)


Bt = np.genfromtxt(f"{D}/base_100Hz.csv", delimiter=",", names=True); tg = Bt["t"]
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(tg, Bt["x_m"], "k", lw=2.5, label="ref (taught)"); ax[1].plot(tg, Bt["y_m"], "k", lw=2.5, label="ref"); ax[2].plot(tg, Bt["yaw_deg"], "k", lw=2.5, label="ref")
X = []
for i, RD in enumerate(sorted(d for d in glob.glob(f"{D}/replay*") if os.path.isdir(d) and glob.glob(f"{d}/replay_base_*.csv"))):
    TS, tr, fw, lf, yw = real(RD); m = (tr > -1) & (tr < tg[-1] + 1); lab = os.path.basename(RD); c = f"C{i}"
    ax[0].plot(tr[m], fw[m], c, label=lab); ax[1].plot(tr[m], lf[m], c, label=lab); ax[2].plot(tr[m], yw[m], c, lw=0.6, label=lab)
    x = np.interp(tg, tr, fw); X.append(x); e = x - Bt["x_m"]
    print(f"{lab}: real vs ref rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.abs(e).max():.3f} m | peaks {x.max():+.3f} / {x.min():+.3f} "
          f"(ref {Bt['x_m'].max():+.3f} / {Bt['x_m'].min():+.3f}) | left {lf[m].min():+.3f}..{lf[m].max():+.3f}")
if len(X) > 1:
    S = np.std(np.array(X), axis=0); print(f"run-to-run spread (std over runs): mean {S.mean():.4f} max {S.max():.4f} m")
ax[0].set_ylabel("forward [m]"); ax[1].set_ylabel("left [m]"); ax[2].set_ylabel("yaw [deg]"); ax[2].set_xlabel("t [s]")
for x_ in ax:
    x_.grid(); x_.legend(fontsize=8)
fig.suptitle(f"{os.path.basename(D)} replays: base actual (Blackfly) vs ref"); fig.tight_layout(); fig.savefig(f"{D}/replays_base_compare.png", dpi=80)
