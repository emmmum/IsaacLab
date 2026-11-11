# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot a replay against its taught session: arm (ref vs encoders) and base (taught Blackfly vs replay odometry vs replay
Blackfly). Usage: plot_replay.py SESSION_DIR REPLAY_DIR REPLAY_TS [CLOCK_OFFSET_S]"""
import json, os, sys
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs_ball  # noqa: E402

D, RD, TS = sys.argv[1:4]; OFF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.047
deg = np.degrees
A = np.genfromtxt(f"{D}/arm_100Hz.csv", delimiter=",", names=True)
Bt = np.genfromtxt(f"{D}/base_100Hz.csv", delimiter=",", names=True)
R = np.genfromtxt(f"{RD}/replay_arm_{TS}.csv", delimiter=",", names=True, dtype=None, encoding=None)
B = np.genfromtxt(f"{RD}/replay_base_{TS}.csv", delimiter=",", names=True, dtype=None, encoding=None)
p = R["phase"] == "play"; ta = R["wall_t"][p] - R["wall_t"][p][0]
tb = B["wall"] - B["wall"][0]; T0 = B["wall"][0]                       # Jetson clock, replay t = 0
a = np.genfromtxt(f"{RD}/blackfly.csv", delimiter=",", names=True, invalid_raise=False); a = a[a["detected"] == 1]
tr = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"]) + OFF - T0    # desktop -> Jetson clock
R0 = cv2.Rodrigues(np.array([a["rx"][0], a["ry"][0], a["rz"][0]]))[0]; p0 = np.array([a["x"][:100].mean(), a["y"][:100].mean(), a["z"][:100].mean()])
fw, lf, yw = [], [], []
for r in a:
    Rr = R0.T @ cv2.Rodrigues(np.array([r["rx"], r["ry"], r["rz"]]))[0]; dp = R0.T @ (np.array([r["x"], r["y"], r["z"]]) - p0)
    fw.append(-dp[0]); lf.append(dp[2]); yw.append(deg(np.arctan2(Rr[0, 2], Rr[2, 2])))   # left = +tag z, yaw CCW+
fw, lf, yw = map(np.array, (fw, lf, yw))
gaps = np.diff(tr); gap_tot = float(gaps[gaps > 0.05].sum())
tg = Bt["t"]; real_x = np.interp(tg, tr, fw)
stats = {"arm_err_rms_deg": [float(deg(np.sqrt(np.mean((R["q" + m][p] - R["cmd" + m][p]) ** 2)))) for m in ("101", "102")],
         "arm_err_max_deg": [float(deg(np.abs(R["q" + m][p] - R["cmd" + m][p]).max())) for m in ("101", "102")],
         "base_real_vs_teach_rms_m": float(np.sqrt(np.mean((real_x - Bt["x_m"]) ** 2))),
         "base_real_vs_teach_max_m": float(np.abs(real_x - Bt["x_m"]).max()),
         "base_odom_vs_teach_rms_m": float(np.sqrt(np.mean((np.interp(tg, tb, B["x"]) - Bt["x_m"]) ** 2))),
         "base_end_teach_real_odom_m": [float(Bt["x_m"][-1]), float(fw[tr <= tg[-1]][-50:].mean()), float(B["x"][-1])],
         "blackfly_gap_s": gap_tot}
try:
    c = cv2.imread(f"{RD}/after_replay/color.jpg"); dep = np.fromfile(f"{RD}/after_replay/depth.raw", np.uint16)[:1280 * 720].reshape(720, 1280)
    o, vis = rs_ball.analyse(c, dep); cv2.imwrite(f"{RD}/ball_after.jpg", vis); stats["ball_after"] = o.get("ball_depth") or o["ball_size"]
except Exception as e:
    stats["ball_after"] = f"not found ({e})"
json.dump(stats, open(f"{RD}/replay_stats.json", "w"), indent=1)
fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax[0].plot(A["t"], A["q101_deg"], "k", lw=2, label="ref 101 (taught)"); ax[0].plot(ta, deg(R["q101"][p]), "C0", label="actual 101")
ax[0].plot(A["t"], A["q102_deg"], "k--", lw=2, label="ref 102 (taught)"); ax[0].plot(ta, deg(R["q102"][p]), "C1", label="actual 102"); ax[0].set_ylabel("arm [deg]")
ax[1].plot(Bt["t"], Bt["x_m"], "k", lw=2, label="ref (taught, Blackfly)"); ax[1].plot(tb, B["x"], "C2", label="replay wheel odometry")
m = (tr > -1) & (tr < tg[-1] + 1); ax[1].plot(tr[m], fw[m], "C3", label="replay actual (Blackfly)"); ax[1].set_ylabel("base forward [m]")
ax[2].plot(Bt["t"], Bt["y_m"], "k", lw=2, label="ref left"); ax[2].plot(tr[m], lf[m], "C3", label="actual left"); ax[2].set_ylabel("base left [m]")
ax[3].plot(Bt["t"], Bt["yaw_deg"], "k", lw=2, label="ref yaw"); ax[3].plot(tr[m], yw[m], "C3", label="actual yaw"); ax[3].set_ylabel("yaw [deg]")
ax[3].set_xlabel("t [s]")
for x in ax:
    x.grid(); x.legend(fontsize=8, loc="upper right")
    for i in np.where(gaps > 0.05)[0]:
        if -1 < tr[i] < tg[-1] + 1: x.axvspan(tr[i], tr[i + 1], color="red", alpha=0.15, lw=0)
fig.suptitle(f"{os.path.basename(D)} replay {TS}: ref vs actual"); fig.tight_layout(); fig.savefig(f"{RD}/replay_vs_ref.png", dpi=80)
print(json.dumps(stats, indent=1))
