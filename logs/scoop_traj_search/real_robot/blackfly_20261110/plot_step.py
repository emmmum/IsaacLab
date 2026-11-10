# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
OFF = 0.0182  # jetson - desktop clock [s]
a = np.genfromtxt("step_tag.csv", delimiter=",", names=True); a = a[a["detected"] == 1]
th = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"])           # camera time mapped to desktop clock (lower envelope)
P = np.c_[a["x"], a["y"], a["z"]]
cmd, vel, pos = [], [], []
bad = 0
for r in (r for r in csv.reader(open("step_odom.csv")) if len(r) > 2):
    try:
        row = (float(r[0]) - OFF, float(r[2]))
    except ValueError:
        bad += 1; continue
    {"cmd": cmd, "vel": vel, "pos": pos}.get(r[1], []).append(row)
print("skipped corrupted odom rows:", bad)
cmd, vel, pos = map(np.array, (cmd, vel, pos))
t_on = cmd[np.r_[True, np.diff(cmd[:, 1]) != 0] & (cmd[:, 1] != 0), 0]   # step onsets (fwd, back)
mv = (th > t_on[0]) & (th < t_on[0] + 1.5)
d = P[mv][-1] - P[mv][0]; u = d / np.linalg.norm(d)                    # motion direction in camera frame
s = (P - P[:20].mean(0)) @ u
dt = np.median(np.diff(th))
v_tag = savgol_filter(s, 21, 2, deriv=1, delta=dt)                       # 0.1 s window
v_raw = np.gradient(savgol_filter(s, 5, 2), th)
print("motion dir (cam frame):", u.round(3), "dt", round(dt*1e3, 2), "ms")
def ref(t):
    i = np.searchsorted(cmd[:, 0], t, side="right") - 1
    return np.where(i >= 0, cmd[np.clip(i, 0, None), 1], 0.0)
fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col")
for k, (t0, lab) in enumerate(zip(t_on, ["forward step +0.4 m/s", "backward step -0.4 m/s"])):
    V = ref(t0 + 0.5); m = (th > t0 - 0.3) & (th < t0 + 2.0); tt = np.linspace(t0 - 0.3, t0 + 2.0, 2000)
    A = ax[0, k]
    A.plot(tt - t0, ref(tt), "k-", lw=2, label="ref (drive_speed cmd)")
    A.plot(th[m] - t0, v_raw[m], color="tab:blue", alpha=0.25, lw=0.8, label="tag 200 Hz (light smoothing)")
    A.plot(th[m] - t0, v_tag[m], color="tab:blue", lw=2, label="tag, SG 0.1 s")
    mo = (vel[:, 0] > t0 - 0.3) & (vel[:, 0] < t0 + 2.0)
    A.plot(vel[mo, 0] - t0, vel[mo, 1], "o-", color="tab:orange", ms=3, lw=1, label="chassis odom vel (50 Hz)")
    # metrics on the tag velocity
    y = v_tag[m] / V; x = th[m] - t0
    t10 = x[np.argmax((x > 0) & (y > 0.1))]; t90 = x[np.argmax((x > 0) & (y > 0.9))]
    pk = y[(x > 0) & (x < 1.2)].max(); tpk = x[(x > 0) & (x < 1.2)][np.argmax(y[(x > 0) & (x < 1.2)])]
    ss = y[(x > 0.8) & (x < 1.2)].mean()
    txt = f"delay(10%) {t10*1e3:.0f} ms\nrise 10-90% {(t90-t10)*1e3:.0f} ms\npeak {pk*abs(V):.3f} m/s ({(pk/ss-1)*100:+.0f}%) at {tpk*1e3:.0f} ms\nsteady {ss*abs(V):.3f} m/s"
    print(lab, txt.replace("\n", " | "))
    A.text(0.98, 0.04 if V > 0 else 0.96, txt, transform=A.transAxes, ha="right", va="bottom" if V > 0 else "top", fontsize=9,
           bbox=dict(fc="white", ec="0.7"))
    A.set_title(lab); A.set_ylabel("velocity [m/s]"); A.grid(alpha=0.3)
    if k == 0: A.legend(loc="upper left", fontsize=8)
    B = ax[1, k]
    B.plot(th[m] - t0, s[m] - s[(th > t0 - 0.3)][0], color="tab:blue", lw=1.5, label="tag displacement")
    mp = (pos[:, 0] > t0 - 0.3) & (pos[:, 0] < t0 + 2.0)
    B.plot(pos[mp, 0] - t0, pos[mp, 1] - pos[mp, 1][0], color="tab:orange", lw=1.5, label="chassis odom x")
    B.set_xlabel("time since command onset [s]"); B.set_ylabel("displacement [m]"); B.grid(alpha=0.3); B.legend(fontsize=8)
fig.suptitle("RoboMaster base: velocity step response (Blackfly S 200 fps AprilTag vs chassis odometry)")
fig.tight_layout(); fig.savefig("step_response.png", dpi=110)
