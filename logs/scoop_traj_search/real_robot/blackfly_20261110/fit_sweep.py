# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
OFF = 0.0182
a = np.genfromtxt("sweep_tag.csv", delimiter=",", names=True); a = a[a["detected"] == 1]
th = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"]); P = np.c_[a["x"], a["y"], a["z"]]
ev = {"cmd": [], "vel": [], "pos": [], "label": []}
for r in csv.reader(open("sweep_odom.csv")):
    if len(r) < 3 or r[1] not in ev: continue
    ev[r[1]].append((float(r[0]) - OFF, r[2]) if r[1] == "label" else (float(r[0]) - OFF, float(r[2])))
cmd, vel, pos = (np.array(ev[k]) for k in ("cmd", "vel", "pos")); labels = ev["label"]
# motion direction: PCA of tag positions, sign so that +x chassis odom == +s
c = P - P.mean(0); u = np.linalg.svd(c, full_matrices=False)[2][0]
s = P @ u; po = np.interp(th, pos[:, 0], pos[:, 1])
if np.corrcoef(s, po)[0, 1] < 0: u, s = -u, -s
print("motion dir", u.round(3), "corr tag/odom", np.corrcoef(s, po)[0, 1].round(4))
dt = np.median(np.diff(th)); v_tag = savgol_filter(s, 21, 2, deriv=1, delta=dt)
def ref(t):
    i = np.searchsorted(cmd[:, 0], t, side="right") - 1
    return np.where(i >= 0, cmd[np.clip(i, 0, None), 1], 0.0)
from scipy.signal import cont2discrete, lfilter
H = 0.001
def sim(p, t, u_of_t):
    tau, wn, ze, K = p; tt = np.arange(t[0], t[-1] + H, H); uu = K * u_of_t(tt - tau)
    bd, ad, _ = cont2discrete(([wn * wn], [1, 2 * ze * wn, wn * wn]), H, method="zoh")
    v = lfilter(bd.ravel(), ad, uu); x = np.cumsum(v) * H
    return np.interp(t, tt, x), np.interp(t, tt, v)
def fit_seg(segs):
    best = None
    for tau in np.arange(0.04, 0.26, 0.005):
        f = lambda q: np.concatenate([sim([tau, *q], t, ref)[0] - y for t, y in segs])
        r = least_squares(f, [12, 0.5, 1.0], bounds=([2, 0.05, 0.5], [60, 2, 1.5]))
        if best is None or r.cost < best[1].cost: best = (tau, r)
    tau, r = best; J = r.jac; sd = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * (r.fun ** 2).mean()))
    return np.r_[tau, r.x], np.r_[0.0025, sd], 1e3 * np.sqrt((r.fun ** 2).mean())
res = []
for k, (t0, lab) in enumerate(labels):
    t_end = labels[k + 1][0] if k + 1 < len(labels) else t0 + 4
    m = (th > t0 - 0.2) & (th < t_end - 0.05); t = th[m]; y = s[m] - s[m][:20].mean()
    px, sd, rms = fit_seg([(t, y)])
    cm = (cmd[:, 0] >= t0) & (cmd[:, 0] < t_end); V = cmd[cm, 1][np.argmax(np.abs(cmd[cm, 1]))]; vm = v_tag[m]; x = t - t0
    pk = (vm * np.sign(V)).max(); tpk = x[np.argmax(vm * np.sign(V))]
    y5 = y * np.sign(V); t5 = x[np.argmax((x > 0) & (y5 > 0.005))]
    po_m = (pos[:, 0] > t0 - 0.2) & (pos[:, 0] < t_end - 0.05); pp = pos[po_m, 1]; disp_odom = abs(pp[-10:].mean() - pp[:5].mean()); disp_tag = abs(y[-40:].mean())
    res.append(dict(label=lab, V=float(V), tau=px[0], wn=px[1], zeta=px[2], K=px[3], sd=sd.tolist(), rms_mm=rms, peak=pk, t_peak=tpk, overshoot=pk / abs(V) - 1, t_5mm=t5,
                    disp_tag=disp_tag, disp_odom=disp_odom, t0=t0, t_end=t_end))
    r = res[-1]; print(f"{lab:14s} tau {r['tau']*1e3:5.0f}ms wn {r['wn']:5.1f} zeta {r['zeta']:.2f} K {r['K']:.3f}  rms {r['rms_mm']:.1f}mm | "
                       f"peak {r['peak']:.3f} ({r['overshoot']*100:+.0f}%) @{r['t_peak']*1e3:.0f}ms  5mm@{r['t_5mm']*1e3:.0f}ms  "
                       f"disp tag {r['disp_tag']:.3f} odom {r['disp_odom']:.3f}")
steps = [r for r in res if r["label"].startswith("step")]
def segs_of(rs):
    out = []
    for r in rs:
        m = (th > r['t0'] - 0.2) & (th < r['t_end'] - 0.05); out.append((th[m], s[m] - s[m][:20].mean()))
    return out
G = {}
for name, rs in [("steps", [r for r in res if r["label"].startswith("step")]), ("traps", [r for r in res if r["label"].startswith("trap")]), ("all", res)]:
    px, sd, rms = fit_seg(segs_of(rs)); G[name] = dict(tau=px[0], wn=px[1], zeta=px[2], K=px[3], rms_mm=rms)
    print(f"GLOBAL {name:5s}: tau {px[0]*1e3:.0f} ms, wn {px[1]:.2f} +- {sd[1]:.2f} rad/s, zeta {px[2]:.3f} +- {sd[2]:.3f}, K {px[3]:.3f}, rms {rms:.1f} mm")
g = type("g", (), {"x": [G["all"][k] for k in ("tau", "wn", "zeta", "K")]})
json.dump(dict(results=res, global_fit=G, motion_dir=u.tolist(), clock_offset=OFF),
          open("sweep_fit.json", "w"), indent=1)
# ---- figure: rows = speeds, cols = step fwd/back, trap fwd/back
fig, ax = plt.subplots(4, 4, figsize=(17, 13))
for r in res:
    kind, V, d = r["label"].split("_"); i = [0.2, 0.4, 0.6, 0.9].index(float(V)); j = (0 if kind == "step" else 2) + (d == "back")
    A = ax[i, j]; t0 = r["t0"]; m = (th > t0 - 0.3) & (th < r["t_end"] - 0.05); tt = np.linspace(t0 - 0.3, r["t_end"] - 0.05, 1500)
    A.plot(tt - t0, ref(tt), "k-", lw=2, label="ref cmd")
    A.plot(th[m] - t0, v_tag[m], color="tab:blue", lw=1.8, label="tag (SG 0.1 s)")
    mo = (vel[:, 0] > t0 - 0.3) & (vel[:, 0] < r["t_end"] - 0.05)
    A.plot(vel[mo, 0] - t0, vel[mo, 1], ".", color="tab:orange", ms=3, label="chassis odom")
    A.plot(th[m] - t0, sim([r["tau"], r["wn"], r["zeta"], r["K"]], th[m], ref)[1], "--", color="tab:red", lw=1.5, label="fit: delay+2nd order")
    A.set_title(f"{kind} {d} {V} m/s   tau {r['tau']*1e3:.0f} ms  wn {r['wn']:.1f}  zeta {r['zeta']:.2f}  OS {r['overshoot']*100:+.0f}%", fontsize=9)
    A.grid(alpha=0.3); A.set_xlim(-0.3, 2.6)
    if j == 0: A.set_ylabel("v [m/s]")
    if i == 3: A.set_xlabel("t since command onset [s]")
ax[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("RoboMaster base velocity response: steps (left) vs trapezoids a=1.5 m/s^2 (right); Blackfly 200 fps tag, 50 Hz drive_speed", fontsize=12)
fig.tight_layout(); fig.savefig("sweep_response.png", dpi=95)
