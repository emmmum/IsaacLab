# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Jetson: capture one onboard D435i color frame (V4L2 /dev/video2) + one depth Z16 frame (/dev/video0), no processing.
Usage: rs_snap.py OUT_DIR  -> OUT_DIR/color.jpg, OUT_DIR/depth.raw (1280x720 uint16 mm)"""
import os, subprocess, sys, time
import cv2
out = sys.argv[1]; os.makedirs(out, exist_ok=True); t0 = time.time()
cap = cv2.VideoCapture(2, cv2.CAP_V4L2); cap.set(3, 1280); cap.set(4, 720)
for _ in range(10): ok, f = cap.read()
cap.release()
cv2.imwrite(os.path.join(out, "color.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
subprocess.run(f"v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat='Z16 ' --stream-mmap --stream-skip=10 "
               f"--stream-count=1 --stream-to={os.path.join(out, 'depth.raw')}", shell=True, capture_output=True)
print(f"snap ok {time.time() - t0:.2f} s, depth {os.path.getsize(os.path.join(out, 'depth.raw'))} B", flush=True)
