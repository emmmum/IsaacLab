# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
TS = open("ts.txt").read().strip(); OFF = float(open("clock_offset.txt").read().split()[0])
A = np.genfromtxt(f"teach_arm_{TS}.csv", delimiter=",", names=True)
ta = A["wall_t"] - OFF
q1, q2 = np.degrees(A["q101"]), np.degrees(A["q102"])
dta = np.median(np.diff(ta)); w = int(round(0.1 / dta)) | 1
qd1, qd2 = (savgol_filter(np.radians(q), w, 2, deriv=1, delta=dta) for q in (q1, q2))
T = np.genfromtxt(f"teach_tag_{TS}.csv", delimiter=",", names=True); T = T[T["detected"] == 1]
tt = T["t_cam_s"] + np.min(T["host_t"] - T["t_cam_s"]); P = np.c_[T["x"], T["y"], T["z"]]
R = np.array([cv2.Rodrigues(np.r_[r])[0] for r in np.c_[T["rx"], T["ry"], T["rz"]]])
yaw = np.degrees(np.unwrap(np.arctan2(R[:, 0, 2], R[:, 2, 2])))          # tag normal heading in camera x-z plane
dtt = np.median(np.diff(tt)); wt = int(round(0.2 / dtt)) | 1
Ps = savgol_filter(P, wt, 2, axis=0); V = savgol_filter(P, wt, 2, deriv=1, delta=dtt, axis=0)
spd = np.linalg.norm(V[:, [0, 2]], axis=1)                                 # planar speed (camera x-z)
# thresholds from the quiet first 1.5 s
qa = ta < ta[0] + 1.5; qt = tt < tt[0] + 1.5
th_q = max(0.05, 6 * np.std(np.r_[qd1[qa], qd2[qa]])); th_v = max(0.02, 6 * np.std(spd[qt]))
mov_a = (np.abs(qd1) > th_q) | (np.abs(qd2) > th_q); mov_t = spd > th_v
def runs(t, m, min_len=0.1):                                               # keep motion runs >= min_len s
    out = []; i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            if t[j - 1] - t[i] >= min_len: out.append((t[i], t[j - 1]))
            i = j
        else: i += 1
    return out
ra, rt = runs(ta, mov_a), runs(tt, mov_t)
allr = sorted(ra + rt)
clusters = []
for r in allr:                                                  # merge runs separated by < 2 s
    if clusters and r[0] - clusters[-1][1] < 2.0: clusters[-1][1] = max(clusters[-1][1], r[1])
    else: clusters.append([r[0], r[1]])
arm_c = [c for c in clusters if any(c[0] <= a0 <= c[1] for a0, _ in ra)]
print("motion clusters (s from arm-log start):", [(round(c[0]-ta[0], 2), round(c[1]-ta[0], 2)) for c in clusters])
demo = [min(c[0] for c in arm_c), max(c[1] for c in arm_c)]
allr = [r for r in allr if demo[0] <= r[0] <= demo[1]]; ra = [r for r in ra if demo[0] <= r[0] <= demo[1]]; rt = [r for r in rt if demo[0] <= r[0] <= demo[1]]
t_start = demo[0] - 1.0; t_end = demo[1] + 1.0
print(f"thresholds: joint {th_q:.3f} rad/s, chassis {th_v:.3f} m/s; arm motion runs {len(ra)}, chassis runs {len(rt)}")
print(f"first motion {min(r[0] for r in allr)-ta[0]:.2f} s after start ({'arm' if ra and min(r[0] for r in ra) <= min([r[0] for r in rt] or [1e18]) else 'chassis'}), "
      f"last motion ends {max(r[1] for r in allr)-ta[0]:.2f} s; window {t_end - t_start:.2f} s")
ma = (ta >= t_start) & (ta <= t_end); mt = (tt >= t_start) & (tt <= t_end)
# unified 100 Hz table
tu = np.arange(t_start, t_end, 0.01); P0 = Ps[mt][0]
cols = {"t": tu - t_start, "q101_deg": np.interp(tu, ta, q1), "q102_deg": np.interp(tu, ta, q2),
        "qd101_rad_s": np.interp(tu, ta, qd1), "qd102_rad_s": np.interp(tu, ta, qd2),
        "tau101_Nm": np.interp(tu, ta, A["tau101"]), "tau102_Nm": np.interp(tu, ta, A["tau102"]),
        "tag_x_m": np.interp(tu, tt, Ps[:, 0]), "tag_y_m": np.interp(tu, tt, Ps[:, 1]), "tag_z_m": np.interp(tu, tt, Ps[:, 2]),
        "chassis_dx_m": np.interp(tu, tt, Ps[:, 0] - P0[0]), "chassis_dz_m": np.interp(tu, tt, Ps[:, 2] - P0[2]),
        "chassis_speed_m_s": np.interp(tu, tt, spd), "tag_yaw_deg": np.interp(tu, tt, yaw - yaw[mt][0])}
gap = np.interp(tu, tt[1:], np.diff(tt)) ; cols["tag_gap_s"] = gap
np.savetxt(f"teach_{TS}_trimmed_100Hz.csv", np.c_[tuple(cols.values())], delimiter=",", header=",".join(cols), comments="", fmt="%.5f")
# raw trimmed copies
import csv as _c
for src, m, tcol, name in [(f"teach_arm_{TS}.csv", ma, ta, "arm"), (f"teach_tag_{TS}.csv", None, None, "tag")]:
    pass
np.savetxt(f"teach_{TS}_arm_raw_trimmed.csv", np.c_[ta[ma] - t_start, q1[ma], q2[ma], A["qd101"][ma], A["qd102"][ma], A["tau101"][ma], A["tau102"][ma]],
           delimiter=",", header="t,q101_deg,q102_deg,qd101_rad_s,qd102_rad_s,tau101_Nm,tau102_Nm", comments="", fmt="%.5f")
np.savetxt(f"teach_{TS}_tag_raw_trimmed.csv", np.c_[tt[mt] - t_start, P[mt], yaw[mt]], delimiter=",",
           header="t,x_m,y_m,z_m,yaw_deg", comments="", fmt="%.5f")
# summary
x = cols
print(f"q101 {x['q101_deg'].min():+.1f}..{x['q101_deg'].max():+.1f} deg, q102 {x['q102_deg'].min():+.1f}..{x['q102_deg'].max():+.1f} deg, "
      f"max |qd| {np.abs(x['qd101_rad_s']).max():.2f}/{np.abs(x['qd102_rad_s']).max():.2f} rad/s")
d = np.hypot(x["chassis_dx_m"], x["chassis_dz_m"])
print(f"chassis: max displacement {d.max()*1e3:.0f} mm, final {d[-1]*1e3:.0f} mm, max speed {x['chassis_speed_m_s'].max():.3f} m/s, "
      f"tag yaw change {x['tag_yaw_deg'].min():+.1f}..{x['tag_yaw_deg'].max():+.1f} deg; tag gaps > 50 ms: {(np.diff(tt[mt]) > 0.05).sum()}")
# figure
fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
t = x["t"]
ax[0].plot(t, x["q101_deg"], label="101 shoulder"); ax[0].plot(t, x["q102_deg"], label="102 elbow"); ax[0].set_ylabel("joint angle [deg]")
ax[1].plot(t, x["qd101_rad_s"], label="101"); ax[1].plot(t, x["qd102_rad_s"], label="102"); ax[1].set_ylabel("joint vel [rad/s]")
ax[2].plot(t, x["chassis_dx_m"] * 1e3, label="tag dx (camera x)"); ax[2].plot(t, x["chassis_dz_m"] * 1e3, label="tag dz (camera depth)")
ax[2].set_ylabel("chassis displacement [mm]")
ax[3].plot(t, x["chassis_speed_m_s"], color="tab:green", label="chassis planar speed"); ax[3].set_ylabel("speed [m/s]")
ax[4].plot(t, x["tag_yaw_deg"], color="tab:purple", label="tag heading change (camera x-z plane)"); ax[4].set_ylabel("heading [deg]"); ax[4].set_xlabel("time [s] (0 = 1 s before first motion)")
for a_ in ax:
    a_.grid(alpha=0.3); a_.legend(fontsize=8, loc="upper right")
    for r in ra: a_.axvspan(r[0] - t_start, r[1] - t_start, color="tab:blue", alpha=0.06)
    for r in rt: a_.axvspan(r[0] - t_start, r[1] - t_start, color="tab:green", alpha=0.08)
    a_.axvline(1.0, color="k", ls=":"); a_.axvline(t[-1] - 1.0, color="k", ls=":")
fig.suptitle(f"Kinesthetic teaching {TS}: arm (gravity-comp, ~175 Hz) + chassis tag (Blackfly 200 fps); shaded = detected motion")
fig.tight_layout(); fig.savefig(f"teach_{TS}.png", dpi=95)
f2, b = plt.subplots(figsize=(6, 6)); sc = b.scatter(x["chassis_dx_m"] * 1e3, x["chassis_dz_m"] * 1e3, c=t, s=4, cmap="viridis")
b.plot(0, 0, "k+", ms=12); b.set_xlabel("camera x [mm]"); b.set_ylabel("camera depth z [mm]"); b.set_aspect("equal"); b.grid(alpha=0.3)
plt.colorbar(sc, label="t [s]"); b.set_title("chassis (tag) path, top view"); f2.tight_layout(); f2.savefig(f"teach_{TS}_path.png", dpi=95)
