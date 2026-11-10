# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import cont2discrete, lfilter, savgol_filter
src = open("fit_sweep.py").read(); exec(src[:src.index("res = []")])           # data loading, ref(), v_tag, s, th
res = json.load(open("sweep_fit.json"))["results"]
H = 0.001
def model_v(p, t, amax_on=True):
    tau, amax, wn, ze, K = p; tt = np.arange(t[0] - 0.5, t[-1] + H, H); uu = K * ref(tt - tau)
    if amax_on:
        r = np.empty_like(uu); r[0] = uu[0]; dv = amax * H
        for i in range(1, len(uu)): r[i] = r[i-1] + min(max(uu[i] - r[i-1], -dv), dv)
        uu = r
    bd, ad, _ = cont2discrete(([wn * wn], [1, 2 * ze * wn, wn * wn]), H, method="zoh")
    return np.interp(t, tt, lfilter(bd.ravel(), ad, uu))
segs = []
for r in res:
    m = (th > r["t0"] - 0.2) & (th < r["t_end"] - 0.05); segs.append((r, th[m], v_tag[m]))
def resid(p, use, amax_on=True):
    return np.concatenate([savgol_filter(model_v(p, t, amax_on), 21, 2) - v for (r, t, v) in use])
fits = {}
for name, sel, amax_on in [("2nd order only (all 16)", lambda r: True, False), ("ratelim+2nd (all 16)", lambda r: True, True),
                           ("ratelim+2nd (steps)", lambda r: r["label"].startswith("step"), True),
                           ("ratelim+2nd (traps)", lambda r: r["label"].startswith("trap"), True)]:
    use = [s_ for s_ in segs if sel(s_[0])]; best = None
    for tau in np.arange(0.03, 0.16, 0.01):
        f = lambda q: resid([tau, *q], use, amax_on)
        x0, lo, hi = ([2.0, 12, 0.4, 1.0], [0.3, 2, 0.02, 0.8], [8, 60, 2, 1.3])
        fr = least_squares(f, x0, bounds=(lo, hi), x_scale=[1, 5, 0.2, 0.1])
        if best is None or fr.cost < best[1].cost: best = (tau, fr)
    tau, fr = best
    for dtau in np.arange(-0.009, 0.0095, 0.003):                       # refine delay
        f = lambda q: resid([tau + dtau, *q], use, amax_on); fr2 = least_squares(f, fr.x, bounds=(lo, hi))
        if fr2.cost < fr.cost: fr, tbest = fr2, tau + dtau
        else: tbest = locals().get("tbest", tau)
    p = [tbest, *fr.x]; rms = np.sqrt((fr.fun ** 2).mean())
    fits[name] = dict(tau=p[0], amax=p[1] if amax_on else None, wn=p[2], zeta=p[3], K=p[4], rms_v=rms)
    print(f"{name:26s}: delay {p[0]*1e3:3.0f} ms  a_max {('%.2f m/s^2' % p[1]) if amax_on else '   --    '}  wn {p[2]:5.2f} rad/s  "
          f"zeta {p[3]:.3f}  K {p[4]:.3f}  rms {rms*1e3:.0f} mm/s", flush=True)
json.dump(fits, open("ratelim_fit.json", "w"), indent=1)
# overlay plot of the global fits
P1 = fits["ratelim+2nd (all 16)"]; P0 = fits["2nd order only (all 16)"]
fig, ax = plt.subplots(4, 4, figsize=(17, 13))
for (r, t, v) in segs:
    kind, V, d = r["label"].split("_"); i = [0.2, 0.4, 0.6, 0.9].index(float(V)); j = (0 if kind == "step" else 2) + (d == "back")
    A = ax[i, j]; t0 = r["t0"]; tt = np.linspace(t[0], t[-1], 1500)
    A.plot(tt - t0, ref(tt), "k-", lw=2, label="ref cmd"); A.plot(t - t0, v, color="tab:blue", lw=1.8, label="tag (real)")
    A.plot(t - t0, savgol_filter(model_v([P1["tau"], P1["amax"], P1["wn"], P1["zeta"], P1["K"]], t), 21, 2), "--", color="tab:red", lw=1.6,
           label="delay+accel limit+2nd order")
    A.plot(t - t0, savgol_filter(model_v([P0["tau"], 1, P0["wn"], P0["zeta"], P0["K"]], t, False), 21, 2), ":", color="tab:green", lw=1.6,
           label="delay+2nd order (no limit)")
    A.set_title(f"{kind} {d} {V} m/s", fontsize=10); A.grid(alpha=0.3); A.set_xlim(-0.2, 2.5)
    if j == 0: A.set_ylabel("v [m/s]")
    if i == 3: A.set_xlabel("t since command onset [s]")
ax[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle(f"One model for all 16 runs: delay {P1['tau']*1e3:.0f} ms, accel limit {P1['amax']:.2f} m/s^2, wn {P1['wn']:.1f} rad/s, "
             f"zeta {P1['zeta']:.2f}, K {P1['K']:.2f}  (rms {P1['rms_v']*1e3:.0f} mm/s)   vs   no limit: delay {P0['tau']*1e3:.0f} ms, "
             f"wn {P0['wn']:.1f}, zeta {P0['zeta']:.2f} (rms {P0['rms_v']*1e3:.0f} mm/s)", fontsize=11)
fig.tight_layout(); fig.savefig("sweep_model.png", dpi=95)
