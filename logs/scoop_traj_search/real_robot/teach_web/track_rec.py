# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Real-time AprilTag (36h11) pose tracking with ROI search + H.264 video (every VIDEO_EVERY-th frame, 720 px wide).
# Usage: track_rec.py DURATION_S EXPOSURE_US OUT.csv TAG_ID VIDEO.mp4   (run with cwd = blackfly_20261110; stop: SIGUSR1)
import queue, subprocess, struct, sys, threading, time, json, numpy as np, cv2
import imageio_ffmpeg
DUR, EXP, OUT = float(sys.argv[1]), sys.argv[2], sys.argv[3]; TAG_ID = int(sys.argv[4]); VIDEO = sys.argv[5]; VIDEO_EVERY = 4
SIZE = 0.09
cal = json.load(open("blackfly_26166162_intrinsics.json")); K = np.array(cal["K"]); D = np.array(cal["dist"])
s = SIZE / 2; OBJ = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX; prm.adaptiveThreshWinSizeMax = 53; prm.minMarkerPerimeterRate = 0.01; prm.errorCorrectionRate = 1.0
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
p = subprocess.Popen(["/usr/bin/python3", "grab.py", "26166162", EXP], stdout=subprocess.PIPE, bufsize=0)
rd = p.stdout
def readn(n):
    b = bytearray();
    while len(b) < n:
        c = rd.read(n - len(b))
        if not c: raise EOFError
        b += c
    return bytes(b)
vq = queue.Queue(maxsize=40); vstate = {"ff": None, "n": 0, "drop": 0}
vt = open(VIDEO[:-4] + "_frames.csv", "w"); vt.write("video_frame,t_cam_s,frame_id\n")
def vwriter():
    while True:
        item = vq.get()
        if item is None: break
        W_, H_, buf = item
        bgr = cv2.resize(cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(H_, W_), cv2.COLOR_BayerRG2RGB), (720, 540), interpolation=cv2.INTER_AREA)
        if vstate["ff"] is None:
            vstate["ff"] = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", "720x540", "-r", str(200 // VIDEO_EVERY), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", VIDEO], stdin=subprocess.PIPE)
        vstate["ff"].stdin.write(bgr.tobytes())
    if vstate["ff"]: vstate["ff"].stdin.close(); vstate["ff"].wait()
vth = threading.Thread(target=vwriter, daemon=True); vth.start()
f = open(OUT, "w"); f.write("t_cam_s,frame_id,host_t,detected,x,y,z,rx,ry,rz,reproj_px,tag_px\n")
roi = None; t0 = time.time(); n = n_det = 0; t_print = t0; ts0 = None; proc_ms = []
import signal, os
stop = [False]; signal.signal(signal.SIGUSR1, lambda *a: stop.__setitem__(0, True))
while time.time() - t0 < DUR and not stop[0]:
    W, H, ts, fid = struct.unpack("<IIQQ", readn(24)); raw = np.frombuffer(readn(W * H), np.uint8).reshape(H, W)
    ts0 = ts if ts0 is None else ts0; tp = time.perf_counter()
    if n % VIDEO_EVERY == 0:
        try: vq.put_nowait((W, H, raw.tobytes())); vt.write(f"{vstate['n']},{(ts-ts0)*1e-9:.6f},{fid}\n"); vstate["n"] += 1
        except queue.Full: vstate["drop"] += 1
    if roi is None: x0, y0, x1, y1 = 0, 0, W, H
    else: x0, y0, x1, y1 = roi
    g = cv2.cvtColor(np.ascontiguousarray(raw[y0:y1, x0:x1]), cv2.COLOR_BayerRG2GRAY)
    cs, ids, _ = det.detectMarkers(g); n += 1; row = None
    if ids is not None and TAG_ID in ids.ravel():
        c = cs[list(ids.ravel()).index(TAG_ID)].reshape(-1, 2) + [x0, y0]
        ok, rv, tv = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        err = np.linalg.norm(cv2.projectPoints(OBJ, rv, tv, K, D)[0].reshape(-1, 2) - c, axis=1).mean()
        side = np.linalg.norm(c - np.roll(c, 1, 0), axis=1).mean(); m = max(60, int(1.2 * side))
        lo, hi = c.min(0), c.max(0)
        roi = (max(0, int(lo[0] - m) & ~1), max(0, int(lo[1] - m) & ~1), min(W, int(hi[0] + m) & ~1), min(H, int(hi[1] + m) & ~1))
        n_det += 1; row = (tv.ravel(), rv.ravel(), err, side)
    else:
        roi = None
    proc_ms.append((time.perf_counter() - tp) * 1e3)
    if row: f.write(f"{(ts-ts0)*1e-9:.6f},{fid},{time.time():.6f},1,{','.join(f'{v:.5f}' for v in row[0])},{','.join(f'{v:.5f}' for v in row[1])},{row[2]:.3f},{row[3]:.1f}\n")
    else: f.write(f"{(ts-ts0)*1e-9:.6f},{fid},{time.time():.6f},0,,,,,,,,\n")
    if time.time() - t_print > 1:
        el = time.time() - t0
        s_ = f"xyz=({row[0][0]:+.3f},{row[0][1]:+.3f},{row[0][2]:.3f}) m tag {row[3]:.0f}px" if row else "tag NOT found"
        print(f"t={el:5.1f}s fps={n/el:6.1f} det={n_det/max(n,1)*100:5.1f}% proc={np.mean(proc_ms[-200:]):.1f}ms  {s_}", flush=True); t_print = time.time()
p.terminate(); f.close(); vq.put(None); vth.join(30); vt.close()
print(f"video: {vstate['n']} frames, dropped {vstate['drop']}", flush=True)
print(f"done: {n} frames in {time.time()-t0:.1f}s, detected {n_det}, cam span {(ts-ts0)*1e-9:.2f}s -> {n/((ts-ts0)*1e-9):.1f} fps")
