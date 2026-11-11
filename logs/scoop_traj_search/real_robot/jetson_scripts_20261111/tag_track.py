# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track AprilTag 36h11 with the onboard D435i color stream (V4L2 /dev/video2, 1280x720) and log the camera pose
in the tag frame. Usage: tag_track.py out.csv duration_s [stopfile]"""
import csv, math, os, sys, time
import numpy as np, cv2
out, dur = sys.argv[1], float(sys.argv[2])
RES = os.environ.get("TAG_RES", "1280")
if RES == "640": K = np.array([[613.213, 0, 328.909], [0, 613.097, 251.656], [0, 0, 1]]); W, H = 640, 480
else: K = np.array([[919.819, 0, 653.364], [0, 919.645, 377.484], [0, 0, 1]]); W, H = 1280, 720
dist = np.zeros(5)
S = float(os.environ.get("TAG_SIZE", "0.120")); h = S / 2
obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)   # aruco corner order: TL, TR, BR, BL
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
prm = cv2.aruco.DetectorParameters_create(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
cap = cv2.VideoCapture(2, cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H); cap.set(cv2.CAP_PROP_FPS, 30)
for _ in range(15): cap.read()
f = open(out, "w", newline=""); w = csv.writer(f)
w.writerow(["wall", "id", "cam_x", "cam_y", "cam_z", "yaw_deg", "pitch_deg", "roll_deg", "dist", "px", "tag_cx", "tag_cy", "tag_cz"])
t0 = time.time(); n = 0; n_det = 0
while time.time() - t0 < dur and not (len(sys.argv) > 3 and os.path.exists(sys.argv[3])):
    ok, img = cap.read(); tw = time.time()
    if not ok: continue
    n += 1
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c, ids, _ = cv2.aruco.detectMarkers(g, d, parameters=prm)
    if ids is None: continue
    for cc, i in zip(c, ids.ravel()):
        okp, rvec, tvec = cv2.solvePnP(obj, cc.reshape(4, 2).astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not okp: continue
        R, _ = cv2.Rodrigues(rvec); Rt = R.T; p = (-Rt @ tvec).ravel()          # camera position in tag frame
        # camera optical axis expressed in the tag frame -> heading angles
        z = Rt[:, 2]; yaw = math.degrees(math.atan2(z[0], -z[2])); pitch = math.degrees(math.asin(-z[1])); roll = math.degrees(math.atan2(Rt[1, 0], Rt[1, 1]))
        w.writerow([f"{tw:.4f}", int(i)] + [f"{x:.5f}" for x in p] + [f"{yaw:.3f}", f"{pitch:.3f}", f"{roll:.3f}", f"{np.linalg.norm(tvec):.4f}", f"{cv2.contourArea(cc.reshape(-1,2))**0.5:.1f}"] + [f"{x:.5f}" for x in tvec.ravel()])
        n_det += 1
f.close(); cap.release()
print(f"frames {n} ({n/dur:.1f} fps), detections {n_det}")
