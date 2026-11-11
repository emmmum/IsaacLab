# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay the Blackfly video of a taught session and of its replay on a common time axis (t = 0: teach START /
replay start), blended 50/50. Usage: overlay_video.py SESSION_DIR REPLAY_DIR [CLOCK_OFFSET_S] -> REPLAY_DIR/overlay.mp4"""
import glob, json, os, subprocess, sys
import cv2
import imageio_ffmpeg
import numpy as np

D, RD = sys.argv[1], sys.argv[2]
OFF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.047          # Jetson clock - desktop clock [s]
FPS = 50


def frames(path):
    """Decode an mp4 into a list of BGR frames (720x540)."""
    p = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 540, 720, 3)


def frame_times(d, t0_desktop):
    """Desktop capture time of each video frame minus t0."""
    a = np.genfromtxt(f"{d}/blackfly.csv", delimiter=",", names=True, invalid_raise=False)
    v = np.genfromtxt(f"{d}/video_frames.csv", delimiter=",", names=True)
    return v["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"]) - t0_desktop


s = json.load(open(f"{D}/summary.json"))
a = np.genfromtxt(f"{D}/blackfly.csv", delimiter=",", names=True, invalid_raise=False); a = a[a["detected"] == 1]
T0_teach = max(s["t_start"], a["t_cam_s"][0] + np.min(a["host_t"] - a["t_cam_s"]))           # same t = 0 as base_100Hz.csv
B = np.genfromtxt(glob.glob(f"{RD}/replay_base_*.csv")[0], delimiter=",", names=True, dtype=None, encoding=None)
T0_replay = B["wall"][0] - OFF                                                                  # chassis start, desktop clock
ft, fr = frame_times(D, T0_teach), frame_times(RD, T0_replay)
vt, vr = frames(f"{D}/video.mp4"), frames(f"{RD}/video.mp4")
ft, fr = ft[:len(vt)], fr[:len(vr)]
t_end = min(s["duration_s"], fr[-1]) + 0.5
out = f"{RD}/overlay.mp4"
ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "720x540",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", out], stdin=subprocess.PIPE)
for t in np.arange(-1.0, t_end, 1.0 / FPS):
    i, j = int(np.argmin(np.abs(ft - t))), int(np.argmin(np.abs(fr - t)))
    img = cv2.addWeighted(vt[i], 0.5, vr[j], 0.5, 0)
    cv2.putText(img, f"t = {t:+.2f} s   50% taught ({os.path.basename(D)}) + 50% replay ({os.path.basename(RD)})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ff.stdin.write(img.tobytes())
ff.stdin.close(); ff.wait()
print(f"wrote {out} ({t_end + 1:.1f} s)")
