# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Auto-collect diverse, static checkerboard views. Usage: collect.py N_TARGET
import subprocess, struct, sys, os, time, numpy as np, cv2
PAT = (11, 7); N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
os.makedirs("views", exist_ok=True)
p = subprocess.Popen(["/usr/bin/python3", "grab.py"], stdout=subprocess.PIPE)
feats, prev, n = [], None, len(os.listdir("views"))
def feat(c, W, H):
    c = c.reshape(PAT[1], PAT[0], 2); tl, tr, bl, br = c[0,0], c[0,-1], c[-1,0], c[-1,-1]
    top, bot = np.linalg.norm(tr-tl), np.linalg.norm(br-bl); lef, rig = np.linalg.norm(bl-tl), np.linalg.norm(br-tr)
    m = c.reshape(-1,2).mean(0)
    return np.array([m[0]/W, m[1]/H, 2*np.sqrt(top*lef)/W, 2*(top-bot)/(top+bot), 2*(lef-rig)/(lef+rig)])
t_last = 0
while n < N:
    hdr = p.stdout.read(8)
    if len(hdr) < 8: break
    W, H = struct.unpack("<II", hdr); raw = np.frombuffer(p.stdout.read(W*H), np.uint8).reshape(H, W)
    g = cv2.cvtColor(raw, cv2.COLOR_BayerRG2GRAY)
    small = cv2.resize(g, (W//2, H//2))
    ok, c = cv2.findChessboardCornersSB(small, PAT, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        prev = None
        if time.time() - t_last > 3: print(f"[{n}/{N}] no board", flush=True); t_last = time.time()
        continue
    c = c * 2
    still = prev is not None and np.abs(c - prev).mean() < 1.0; prev = c
    f = feat(c, W, H)
    d = min([np.linalg.norm(f - q) for q in feats], default=9)
    if still and d > 0.10:
        ok2, cf = cv2.findChessboardCornersSB(g, PAT, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok2: continue
        cv2.imwrite(f"views/v{n:02d}.png", g); np.save(f"views/v{n:02d}.npy", cf.reshape(-1,2))
        feats.append(f); n += 1
        print(f"[{n}/{N}] saved  centre=({f[0]:.2f},{f[1]:.2f}) size={f[2]:.2f} tilt=({f[3]:+.2f},{f[4]:+.2f})", flush=True)
        subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/camera-shutter.oga"])
    elif time.time() - t_last > 1.5:
        print(f"[{n}/{N}] board seen, {'moving' if not still else 'too similar (d=%.2f)' % d}", flush=True); t_last = time.time()
p.terminate()
subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"]); print("done", n)
