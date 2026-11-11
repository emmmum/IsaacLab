# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; RUNS = sys.argv[2:]
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
Tt = np.genfromtxt(D + "/teach_tag_20261111_090229.csv", delimiter=",", names=True); A = np.genfromtxt(D + "/teach_arm_20261111_090229.csv", delimiter=",", names=True)
tt = Tt["wall"] - A["wall_t"][0] - 25.5 - 0.1; x = Tt["dist"][tt < 1].mean() - Tt["dist"]; m = (tt > -1) & (tt < 10); ax[1].plot(tt[m], x[m], "k.", ms=3, label="teach (tag)")
ref = np.genfromtxt(D + "/teach_20261111_090229_chassis_100Hz.csv", delimiter=",", names=True); ax[0].plot(ref["t"], ref["x_m"], "k", lw=2, label="ref")
for TS, c in zip(RUNS, ("C0", "C1", "C2", "C3")):
    B = np.genfromtxt(D + f"/replay_base_{TS}.csv", delimiter=",", names=True, dtype=None, encoding=None); t0 = B["wall"][0]; tb = (B["wall"] - t0) * 0.2
    e = B["x_ref"] - B["x"]
    print(TS, "base err rms %.3f max %.3f m | vx_cmd max %.2f | y max %.3f | yaw max %.2f deg" % (np.sqrt((e ** 2).mean()), np.abs(e).max(), np.abs(B["vx_cmd"]).max(), np.abs(B["y"]).max(), np.abs(B["yaw"] - B["yaw"][0]).max()))
    ax[0].plot(tb, B["x"], c, label="odom " + TS[-6:])
    T = np.genfromtxt(D + f"/replay_tag_{TS}.csv", delimiter=",", names=True); tq = (T["wall"] - 0.1 - t0) * 0.2; xq = T["dist"][tq < 0.9].mean() - T["dist"]; mm = (tq > -1) & (tq < 10)
    ax[1].plot(tq[mm], xq[mm], c + ".", ms=2, label="replay tag " + TS[-6:])
    R = np.genfromtxt(D + f"/replay_arm_{TS}.csv", delimiter=",", names=True, dtype=None, encoding=None); p = R["phase"] == "play"
    print("   arm err rms %.2f / %.2f deg, tau101 max %.2f" % tuple([np.degrees(np.sqrt(((R["q" + k][p] - R["cmd" + k][p]) ** 2).mean())) for k in ("101", "102")] + [np.abs(R["tau101"][p]).max()]))
for a in ax: a.grid(); a.legend(); a.set_ylabel("forward [m]")
ax[1].set_xlabel("teach time [s] (replayed at 0.2x)"); ax[0].set_title("chassis: reference vs wheel odometry"); ax[1].set_title("AprilTag forward travel: teach vs replays")
fig.tight_layout(); fig.savefig(D + "/replay_base_x0.2.png", dpi=85)
