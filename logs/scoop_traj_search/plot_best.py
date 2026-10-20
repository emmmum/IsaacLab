# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot and animate the recorded best open-loop scoop run (side view, x-z plane)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_run")
TITLE = sys.argv[2] if len(sys.argv) > 2 else "Open-loop scoop trajectory: v = 0.8 m/s, lift at 0.8 s (1 rad/s), stop at 1.3 s"
ANIM_TITLE = sys.argv[3] if len(sys.argv) > 3 else "Open-loop scoop trajectory (no uncertainty) -- 96% success over 150 runs; this run: success"
T_LIFT_ARG = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
T_BRAKE_ARG = float(sys.argv[5]) if len(sys.argv) > 5 else 1.3
d = dict(np.load(os.path.join(run_dir, "trajectory_record.npz")))
t = d["t"]

# palette (light): categorical slots + ink tokens
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
C_SH, C_EL, C_BALL, C_LIP, C_BASE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})

T_LIFT, T_BRAKE = T_LIFT_ARG, T_BRAKE_ARG
t_contact = t[np.argmax(d["in_net"] > 0)]
events = [(t_contact, "ball touches net"), (T_LIFT, "arm lift starts"), (T_BRAKE, "drive stops")]
in_net = d["in_net"] > 0


def shade_in_net(ax):
    # contiguous in-net spans
    edges = np.flatnonzero(np.diff(np.r_[0, in_net.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[s], t[e - 1], color=C_BALL, alpha=0.08, lw=0)


def draw_events(ax, label=False):
    # labels sit in the empty band under the speed curve, staggered so close events do not collide
    placement = [(0.55, "right", -0.015), (0.38, "left", 0.015), (0.55, "left", 0.015)]
    for (x, name), (y, ha, dx) in zip(events, placement):
        ax.axvline(x, color=MUTED, lw=1, ls=":")
        if label:
            ax.text(x + dx, y, name, transform=ax.get_xaxis_transform(), ha=ha, va="center", fontsize=8, color=INK2)


def ts_panels(axes):
    a0, a1, a2 = axes
    a0.plot(t, d["vx_cmd"], color=C_BASE, ls="--", lw=1.5, label="command")
    a0.plot(t, d["vx"], color=C_BASE, label="actual")
    a0.set_ylabel("base speed [m/s]")
    a0.set_title("Base forward speed")
    a0.legend(loc="upper right", frameon=False, fontsize=8)
    a1.plot(t, np.degrees(d["sh_target"]), color=C_SH, ls="--", lw=1.5)
    a1.plot(t, np.degrees(d["sh"]), color=C_SH, label="shoulder")
    a1.plot(t, np.degrees(d["el_target"]), color=C_EL, ls="--", lw=1.5)
    a1.plot(t, np.degrees(d["el"]), color=C_EL, label="elbow")
    a1.set_ylabel("joint angle [deg]")
    a1.set_title("Arm joints (solid = actual, dashed = target; + = head up)")
    a1.legend(loc="upper left", frameon=False, fontsize=8)
    a2.plot(t, d["ball_z"] - 0.0325, color=C_BALL, label="ball clearance above ground")
    a2.plot(t, d["lip_z"], color=C_LIP, label="net lip height")
    a2.set_ylabel("height [m]")
    a2.set_xlabel("time [s]")
    a2.set_title("Ball and net lip height (green band = ball in net)")
    a2.legend(loc="upper left", frameon=False, fontsize=8)
    for i, ax in enumerate(axes):
        shade_in_net(ax)
        draw_events(ax, label=(i == 0))
        ax.set_xlim(t[0], t[-1])


# ---------- static summary figure ----------
fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, constrained_layout=True)
ts_panels(axes)
fig.suptitle(TITLE, x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(run_dir, "trajectory_timeseries.png"), dpi=150)
plt.close(fig)

# ---------- animation ----------
fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.18, left=0.05, right=0.98, top=0.9, bottom=0.09)
ax_s = fig.add_subplot(gs[:, 0])
ts_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
for ax in ts_axes[1:]:
    ax.sharex(ts_axes[0])
ts_panels(ts_axes)
for ax in ts_axes[:2]:
    ax.tick_params(labelbottom=False)
cursors = [ax.axvline(t[0], color=INK, lw=1.2) for ax in ts_axes]

ax_s.set_aspect("equal")
ax_s.set_xlim(-0.35, 1.75)
ax_s.set_ylim(-0.05, 0.95)
ax_s.set_title("Side view (schematic chassis; arm, net and ball from simulation)")
ax_s.set_xlabel("x [m]")
ax_s.set_ylabel("z [m]")
ax_s.axhline(0, color=INK2, lw=2)
ax_s.grid(True)

chassis = Rectangle((0, 0.04), 0.32, 0.09, fc="#d9d8d4", ec=INK2, lw=1.2)
ax_s.add_patch(chassis)
wheels = [Circle((0, 0.05), 0.05, fc="#bdbcb8", ec=INK2, lw=1.2) for _ in range(2)]
for w in wheels:
    ax_s.add_patch(w)
(mast,) = ax_s.plot([], [], color=INK2, lw=4, solid_capstyle="round")
(upper,) = ax_s.plot([], [], color=C_SH, lw=5, solid_capstyle="round")
(stick,) = ax_s.plot([], [], color=C_EL, lw=3, solid_capstyle="round")
(head,) = ax_s.plot([], [], color=C_LIP, lw=6, solid_capstyle="round", alpha=0.85)
joint_sh = Circle((0, 0), 0.018, fc=SURF, ec=C_SH, lw=2, zorder=5)
joint_el = Circle((0, 0), 0.018, fc=SURF, ec=C_EL, lw=2, zorder=5)
ball = Circle((0, 0), 0.0325, fc=C_BALL, ec=INK2, lw=1, zorder=6)
for p in (joint_sh, joint_el, ball):
    ax_s.add_patch(p)
(ball_trail,) = ax_s.plot([], [], color=C_BALL, lw=1, ls=":", alpha=0.8)
txt = ax_s.text(0.02, 0.97, "", transform=ax_s.transAxes, va="top", fontsize=10, family="monospace", color=INK)
fig.suptitle(ANIM_TITLE,
             x=0.05, ha="left", fontsize=12, fontweight="bold")


def update(i):
    bx = d["base_x"][i]
    sx, sz = d["shoulder_x"][i], d["shoulder_z"][i]
    ex, ez = d["elbow_x"][i], d["elbow_z"][i]
    px, pz = d["pocket_x"][i], d["pocket_z"][i]
    lx, lz = d["lip_x"][i], d["lip_z"][i]
    chassis.set_x(bx - 0.16)
    wheels[0].center = (bx - 0.1, 0.05)
    wheels[1].center = (bx + 0.1, 0.05)
    mast.set_data([sx, sx], [0.13, sz])
    upper.set_data([sx, ex], [sz, ez])
    stick.set_data([ex, lx], [ez, lz])
    # net head: from ~0.1 m behind the pocket center to the lip, along the stick
    u = np.array([lx - ex, lz - ez]) / np.hypot(lx - ex, lz - ez)
    hb = np.array([px, pz]) - 0.1 * u
    head.set_data([hb[0], lx], [hb[1], lz])
    joint_sh.center = (sx, sz)
    joint_el.center = (ex, ez)
    ball.center = (d["ball_x"][i], d["ball_z"][i])
    ball_trail.set_data(d["ball_x"][: i + 1], d["ball_z"][: i + 1])
    state = "IN NET" if in_net[i] else ("on ground" if d["ball_z"][i] - 0.0325 < 0.003 else "in air")
    txt.set_text(
        f"t = {t[i]:4.2f} s   speed {d['vx'][i]:4.2f} m/s\n"
        f"shoulder {np.degrees(d['sh'][i]):+6.1f} deg  elbow {np.degrees(d['el'][i]):+6.1f} deg\n"
        f"ball: {state}, clearance {max(d['ball_z'][i] - 0.0325, 0):.3f} m"
    )
    for c in cursors:
        c.set_xdata([t[i], t[i]])
    return []


fps = 30  # sim runs at 60 Hz -> 0.5x real time
writer_mp4 = FFMpegWriter(fps=fps, bitrate=3000)
with writer_mp4.saving(fig, os.path.join(run_dir, "trajectory_animation.mp4"), dpi=110):
    for i in range(len(t)):
        update(i)
        writer_mp4.grab_frame()
# lighter GIF: every 2nd step at 15 fps (also 0.5x real time)
writer_gif = PillowWriter(fps=15)
with writer_gif.saving(fig, os.path.join(run_dir, "trajectory_animation.gif"), dpi=70):
    for i in range(0, len(t), 2):
        update(i)
        writer_gif.grab_frame()
# a few key frames as stills
for name, tt in [("frame_contact", t_contact), ("frame_lift", 1.0), ("frame_end", t[-1])]:
    update(int(np.argmin(np.abs(t - tt))))
    fig.savefig(os.path.join(run_dir, f"{name}.png"), dpi=110)
print("PLOT done", os.listdir(run_dir))
