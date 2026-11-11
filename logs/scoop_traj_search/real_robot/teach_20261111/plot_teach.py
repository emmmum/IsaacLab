# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = sys.argv[1]; ts = sys.argv[2]
A = np.genfromtxt(f"{d}/teach_arm_{ts}.csv", delimiter=",", names=True)
T = np.genfromtxt(f"{d}/teach_tag_{ts}.csv", delimiter=",", names=True)
t0 = A["wall_t"][0]; ta = A["wall_t"] - t0; tt = T["wall"] - t0
print(f"arm {ta[-1]:.1f} s, {len(ta)/ta[-1]:.0f} Hz | tag {len(tt)} det, span {tt[0]:.1f}-{tt[-1]:.1f} s, max gap {np.diff(tt).max():.2f} s")
for k in ("q101", "q102"): print(k, "deg range", np.degrees(A[k]).min().round(1), np.degrees(A[k]).max().round(1))
print("tag_cz range", T["tag_cz"].min().round(3), T["tag_cz"].max().round(3), " tag_cx range", T["tag_cx"].min().round(3), T["tag_cx"].max().round(3))
gaps = np.where(np.diff(tt) > 0.3)[0]; print("gaps>0.3s:", [(round(tt[i],1), round(tt[i+1]-tt[i],2)) for i in gaps][:20])
fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
ax[0].plot(ta, np.degrees(A["q101"]), label="101 shoulder"); ax[0].plot(ta, np.degrees(A["q102"]), label="102 elbow")
ax[0].plot(ta, np.degrees(A["q102"] - A["q101"]), "--", label="stick pitch (q102-q101)"); ax[0].set_ylabel("deg"); ax[0].legend(); ax[0].grid()
ax[1].plot(ta, A["tau101"], label="tau101"); ax[1].plot(ta, A["tau102"], label="tau102"); ax[1].plot(ta, A["ff101"], ":", label="ff101"); ax[1].plot(ta, A["ff102"], ":", label="ff102"); ax[1].set_ylabel("N m"); ax[1].legend(); ax[1].grid()
ax[2].plot(tt, T["tag_cz"][0] - T["tag_cz"], ".-", ms=3, label="forward travel (tag_cz0 - tag_cz)"); ax[2].plot(tt, T["dist"][0] - T["dist"], ".", ms=2, label="dist0 - dist")
ax[2].set_ylabel("m"); ax[2].legend(); ax[2].grid()
ax[3].plot(tt, T["tag_cx"], ".-", ms=3, label="tag_cx (tag right of cam)"); ax[3].set_ylabel("m"); ax[3].legend(); ax[3].grid(); ax[3].set_xlabel("t [s] from arm log start")
fig.suptitle(f"Teach {ts}"); fig.tight_layout(); fig.savefig(f"{d}/teach_{ts}.png", dpi=90)
