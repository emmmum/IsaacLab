# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob, json, numpy as np, cv2
import sys
PAT = (11, 7); SQX, SQY = float(sys.argv[1]), float(sys.argv[2]); SQ = SQX
obj = np.zeros((PAT[0]*PAT[1], 3), np.float32); obj[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2) * [SQX, SQY]
files = sorted(glob.glob("views/v*.npy")); img = [np.load(f).astype(np.float32) for f in files]
W, H = 1440, 1080
def run(imgs, flags):
    return cv2.calibrateCameraExtended([obj]*len(imgs), [c.reshape(-1,1,2) for c in imgs], (W, H), np.array([[2290.,0,720],[0,2290.,540],[0,0,1]]), None, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
flags = cv2.CALIB_FIX_K3 | (cv2.CALIB_FIX_ASPECT_RATIO if "fixar" in sys.argv else 0)
rms, K, D, rv, tv, sdi, sde, per = run(img, flags)
per = per.ravel()
print(f"{len(img)} views, rms {rms:.3f} px; per-view max {per.max():.3f} ({files[per.argmax()]})")
keep = per < max(3*np.median(per), 0.5)
if not keep.all():
    img = [c for c, k in zip(img, keep) if k]; files = [f for f, k in zip(files, keep) if k]
    rms, K, D, rv, tv, sdi, sde, per = run(img, flags); print(f"refit without outliers: {len(img)} views, rms {rms:.3f} px")
sd = sdi.ravel()
print("fx fy cx cy:", K[0,0].round(2), K[1,1].round(2), K[0,2].round(2), K[1,2].round(2), " +-", sd[:4].round(2))
print("dist k1 k2 p1 p2:", D.ravel()[:4].round(5), " +-", sd[4:8].round(5))
print(f"HFOV {np.degrees(2*np.arctan(W/2/K[0,0])):.1f} deg, VFOV {np.degrees(2*np.arctan(H/2/K[1,1])):.1f} deg")
dist = [np.linalg.norm(t) for t in tv]; print(f"board distance {min(dist):.2f}-{max(dist):.2f} m")
json.dump({"serial": "26166162", "width": W, "height": H, "K": K.tolist(), "dist": D.ravel().tolist(), "rms_px": rms,
           "n_views": len(img), "square_m": [SQX, SQY], "pattern": PAT}, open("blackfly_26166162_intrinsics.json", "w"), indent=2)
cov = np.zeros((H, W), np.uint8)
for c in img: cv2.fillConvexPoly(cov, cv2.convexHull(c.astype(np.int32)), 1)
print(f"image area covered by at least one board: {cov.mean()*100:.0f}%")
