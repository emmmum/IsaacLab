# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phone web page for teaching the lacrosse scoop: START, then SUCCESS or FAIL.

START  : RealSense snapshot on the Jetson (rs_snap.py: color + raw depth, no processing), Jetson clock-offset probe,
         Blackfly chassis-tag tracking + H.264 video on the desktop (track_rec.py). The arm encoders are logged
         continuously on the Jetson by grav_comp_rec.py (file named in ~/robomaster_test/ARM_LIVE).
SUCCESS / FAIL : stop the Blackfly, wait for the arm log to be flushed past the end, cut the arm slice, fetch the
         snapshot and locate the ball on the desktop (rs_ball.py), convert the tag poses to mobile-base motion
         (forward / left / yaw in the start-pose frame), write 100 Hz replay files + plot, store the outcome.

Run on the desktop:  sg flirimaging -c "env_isaaclab/bin/python teach_web.py [port]"   -> http://<desktop-ip>:<port>/
"""
import http.server, json, os, signal, subprocess, sys, threading, time, traceback
import urllib.parse
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rs_ball  # noqa: E402

SESS = os.path.join(HERE, "sessions")
BF_DIR = os.path.join(HERE, "..", "blackfly_20261110")
PY = sys.executable
JS = os.path.join(HERE, "js.sh")
JHOME = "/home/jetson/robomaster_test"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
EXPOSURE_US = "2000"

lock = threading.Lock()
state = {"phase": "idle", "msg": "ready", "session": None, "busy": False, "log": []}
rec = {}


def say(msg):
    with lock:
        state["msg"] = msg
        state["log"] = (state["log"] + [time.strftime("%H:%M:%S ") + msg])[-12:]
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def jetson(cmd, timeout=60):
    r = subprocess.run([JS, cmd], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"jetson cmd failed ({r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout


def clock_offset(n=5):
    """Jetson time - desktop time [s], from the probe with the smallest round trip."""
    best = None
    for _ in range(n):
        a = time.time()
        j = float(jetson("python3 -c 'import time;print(time.time())'", 15))
        b = time.time()
        if best is None or b - a < best[1]:
            best = (j - (a + b) / 2, b - a)
    return best


def do_start():
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        d = os.path.join(SESS, ts)
        os.makedirs(d)
        with lock:
            state.update(session=ts, phase="starting")
        say("RealSense snapshot ...")
        jetson(f"cd {JHOME} && python3 rs_snap.py teach_sessions/{ts}", 30)
        off, rtt = clock_offset()
        arm_file = jetson(f"cat {JHOME}/ARM_LIVE").decode().strip()
        if not jetson("pgrep -f '^python3 grav_comp_rec' || true").decode().strip():
            raise RuntimeError("grav_comp_rec.py is not running on the Jetson (no arm encoder log)")
        say("starting Blackfly ...")
        f_log = open(os.path.join(d, "blackfly.log"), "w")
        p = subprocess.Popen([PY, os.path.join(HERE, "track_rec.py"), "3600", EXPOSURE_US, os.path.join(d, "blackfly.csv"), "0",
                              os.path.join(d, "video.mp4")], cwd=BF_DIR, stdout=f_log, stderr=subprocess.STDOUT)
        t_wait = time.time()
        while time.time() - t_wait < 15:
            if p.poll() is not None:
                raise RuntimeError("Blackfly tracker exited: " + open(os.path.join(d, "blackfly.log")).read()[-300:])
            if "fps=" in open(os.path.join(d, "blackfly.log")).read():
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Blackfly gave no frames in 15 s")
        bl = open(os.path.join(d, "blackfly.log")).read().strip().splitlines()[-1]
        rec.update(dir=d, ts=ts, proc=p, f_log=f_log, t_start=time.time(), off0=off, rtt0=rtt, arm_file=arm_file)
        with lock:
            state["phase"] = "recording"
        say(f"RECORDING  (clock {off * 1e3:+.0f} ms) | {bl.split('det=')[-1]}")
    except Exception as e:
        traceback.print_exc()
        if rec.get("proc"):
            rec["proc"].send_signal(signal.SIGUSR1)
        rec.clear()
        with lock:
            state["phase"] = "idle"
        say(f"START FAILED: {e}")
    finally:
        with lock:
            state["busy"] = False


def fetch_arm(d, arm_file, tj0, tj1):
    """Cut [tj0, tj1] (Jetson time) out of the continuous arm log; waits until the log is flushed past tj1."""
    t_w = time.time()
    while True:
        last = jetson(f"tail -c 400 {JHOME}/{arm_file} | tail -2 | head -1 | cut -d, -f1").decode().strip()
        if last and float(last) >= tj1:
            break
        if time.time() - t_w > 20:
            raise RuntimeError(f"arm log not flushed past the end (last {last}, need {tj1:.2f})")
        time.sleep(1.0)
    arm_csv = jetson(f"cd {JHOME} && awk -F, 'NR==1 || ($1 >= {tj0:.3f} && $1 <= {tj1:.3f})' {arm_file}", 60)
    open(os.path.join(d, "arm_raw.csv"), "wb").write(arm_csv)


def ball_from_snapshot(d, ts):
    open(os.path.join(d, "rs_color.jpg"), "wb").write(jetson(f"cat {JHOME}/teach_sessions/{ts}/color.jpg"))
    open(os.path.join(d, "rs_depth.raw"), "wb").write(jetson(f"cat {JHOME}/teach_sessions/{ts}/depth.raw"))
    color = cv2.imread(os.path.join(d, "rs_color.jpg"))
    dep = np.fromfile(os.path.join(d, "rs_depth.raw"), np.uint16)[:1280 * 720].reshape(720, 1280)
    out, vis = rs_ball.analyse(color, dep)
    cv2.imwrite(os.path.join(d, "ball_photo.jpg"), vis)
    json.dump(out, open(os.path.join(d, "ball_floor.json"), "w"), indent=1)
    return out


def do_stop(outcome):
    try:
        d, p, ts = rec["dir"], rec["proc"], rec["ts"]
        t_stop = time.time()
        with lock:
            state["phase"] = "processing"
        say(f"{outcome.upper()} - stopping Blackfly ...")
        p.send_signal(signal.SIGUSR1)
        p.wait(40)
        rec["f_log"].close()
        off, rtt = clock_offset()
        offm = 0.5 * (off + rec["off0"])
        say("waiting for the arm log ...")
        fetch_arm(d, rec["arm_file"], rec["t_start"] + offm - 0.5, t_stop + offm + 0.5)
        say("ball from RealSense snapshot ...")
        try:
            ball = ball_from_snapshot(d, ts)
        except Exception as e:
            ball = {"error": str(e)}
        say("processing ...")
        summary = process(d, rec["t_start"], t_stop, offm)
        summary.update(outcome=outcome, t_start=rec["t_start"], t_stop=t_stop, clock_offset_s=[rec["off0"], off],
                       rtt_s=[rec["rtt0"], rtt], ball=ball)
        json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
        with lock:
            state["phase"] = "idle"
            state["result"] = summary
        say(f"DONE {ts}: {outcome.upper()}")
    except Exception as e:
        traceback.print_exc()
        with lock:
            state["phase"] = "idle"
        say(f"STOP/PROCESS FAILED: {e}")
    finally:
        rec.clear()
        with lock:
            state["busy"] = False


def process(d, t_start, t_stop, off):
    # --- base from the Blackfly chassis tag: pose in the start-pose frame (tag x = car backward, y = down, z = car left)
    a = np.genfromtxt(os.path.join(d, "blackfly.csv"), delimiter=",", names=True, invalid_raise=False)
    n_all = len(a)
    a = a[a["detected"] == 1]
    if len(a) < 50:
        raise RuntimeError(f"chassis tag detected in only {len(a)}/{n_all} frames")
    tcap = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"])             # desktop time of capture
    k0 = min(len(a), 100)
    R0 = cv2.Rodrigues(np.array([a["rx"][0], a["ry"][0], a["rz"][0]]))[0]
    p0 = np.array([a["x"][:k0].mean(), a["y"][:k0].mean(), a["z"][:k0].mean()])
    fwd, left, yaw = [], [], []
    for r in a:
        R = cv2.Rodrigues(np.array([r["rx"], r["ry"], r["rz"]]))[0]
        dp = R0.T @ (np.array([r["x"], r["y"], r["z"]]) - p0)
        Rr = R0.T @ R
        fwd.append(-dp[0]); left.append(dp[2]); yaw.append(np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])))   # left = +tag z, yaw CCW+
    fwd, left, yaw = np.array(fwd), np.array(left), np.array(yaw)
    T0 = max(t_start, tcap[0])
    tg = np.arange(0.0, min(t_stop, tcap[-1]) - T0, 0.01)
    tb = tcap - T0

    # linear interpolation onto the grid, Gaussian smoothing on the uniform grid; grid points inside tag-loss gaps are masked
    def gsmooth(x, sig_s):
        k = np.exp(-0.5 * (np.arange(-4 * sig_s, 4 * sig_s + 1e-9, 0.01) / sig_s) ** 2); k /= k.sum()
        return np.convolve(np.pad(x, len(k) // 2, mode="edge"), k, "valid")

    X, Y, W = (gsmooth(np.interp(tg, tb, v), s) for v, s in ((fwd, 0.02), (left, 0.02), (yaw, 0.03)))
    VX, VY = np.gradient(X, tg), np.gradient(Y, tg)
    VXs = gsmooth(VX, 0.03); AX = np.gradient(VXs, tg)
    gaps = np.diff(tb); max_gap = float(gaps.max()) if len(gaps) else 0.0
    bad = np.zeros(len(tg), bool)
    for i in np.where(gaps > 0.05)[0]:
        bad |= (tg > tb[i] - 0.05) & (tg < tb[i + 1] + 0.05)
    gap_total = float(gaps[gaps > 0.05].sum())
    # --- arm (Jetson time -> desktop time)
    A = np.genfromtxt(os.path.join(d, "arm_raw.csv"), delimiter=",", names=True)
    if A.size < 10:
        raise RuntimeError("no arm encoder samples in the recording window")
    ta = A["wall_t"] - off - T0
    q1, q2 = np.interp(tg, ta, np.degrees(A["q101"])), np.interp(tg, ta, np.degrees(A["q102"]))
    np.savetxt(os.path.join(d, "arm_100Hz.csv"), np.c_[tg, q1, q2], delimiter=",", header="t,q101_deg,q102_deg", comments="", fmt="%.5f")
    np.savetxt(os.path.join(d, "base_100Hz.csv"), np.c_[tg, X, VX, Y, VY, W], delimiter=",",
               header="t,x_m,vx_m_s,y_m,vy_m_s,yaw_deg", comments="", fmt="%.5f")
    np.savetxt(os.path.join(d, "base_raw.csv"), np.c_[tb, fwd, left, yaw], delimiter=",", header="t,fwd_m,left_m,yaw_deg", comments="", fmt="%.5f")
    fig, ax = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    ax[0].plot(tg, q1, label="101 shoulder"); ax[0].plot(tg, q2, label="102 elbow"); ax[0].set_ylabel("arm [deg]")
    ax[1].plot(tb, fwd, ".", ms=1, color="0.7"); ax[1].plot(tg, X, label="forward"); ax[1].plot(tg, Y, label="left"); ax[1].set_ylabel("base [m]")
    ax[2].plot(tg, VXs, label="vx"); ax[2].plot(tg, AX / 10, label="ax / 10"); ax[2].set_ylabel("m/s, m/s2 /10")
    ax[3].plot(tg, W, label="yaw"); ax[3].set_ylabel("yaw [deg]"); ax[3].set_xlabel("t [s] since START")
    for x_ in ax:
        x_.grid(); x_.legend(loc="upper right")
        for i in np.where(gaps > 0.05)[0]:
            x_.axvspan(tb[i], tb[i + 1], color="red", alpha=0.15, lw=0)
    fig.suptitle(os.path.basename(d)); fig.tight_layout(); fig.savefig(os.path.join(d, "plot.png"), dpi=70); plt.close(fig)
    good = ~bad
    return dict(duration_s=float(tg[-1]), blackfly_frames=int(n_all), tag_detected=int(len(a)), tag_max_gap_s=max_gap,
                tag_gap_total_s=gap_total,
                base_fwd_range_m=[float(X.min()), float(X.max())], base_end_m=[float(X[-1]), float(Y[-1])],
                base_vx_max_m_s=float(np.abs(VXs[good]).max()) if good.any() else float("nan"),
                base_ax_max_m_s2=float(np.abs(AX[good]).max()) if good.any() else float("nan"),
                yaw_range_deg=[float(W.min()), float(W.max())],
                arm_q101_range_deg=[float(q1.min()), float(q1.max())], arm_q102_range_deg=[float(q2.min()), float(q2.max())])


def session_list():
    out = []
    for ts in sorted(os.listdir(SESS), reverse=True):
        f = os.path.join(SESS, ts, "summary.json")
        if not os.path.isfile(f):
            out.append(dict(ts=ts, outcome="incomplete")); continue
        s = json.load(open(f)); b = s.get("ball", {})
        bd = b.get("ball_depth") or b.get("ball_size")
        out.append(dict(ts=ts, outcome=s.get("outcome", "?"), dur=s["duration_s"], vx=s["base_vx_max_m_s"],
                        gap=s.get("tag_gap_total_s", 0), ball=None if not bd else [bd["forward"], bd["left"]]))
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scoop Teach</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#ddd;--go:#1f5fa8;--ok:#1f7a4d;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eee;--mut:#a3a39c;--card:#22221f;--line:#3a3a36}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif;padding:16px;max-width:640px;margin:auto}
h1{font-size:20px;margin:4px 0 12px}h2{font-size:16px;margin:0 0 6px}.row{display:flex;gap:12px;margin-top:12px}
button{flex:1;font-size:22px;font-weight:600;padding:26px 0;border:0;border-radius:14px;color:#fff}
#go{background:var(--go);width:100%}#ok{background:var(--ok)}#bad{background:var(--bad)}button:disabled{opacity:.3}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-top:14px}
#ph{font-size:24px;font-weight:700}#msg{color:var(--mut);margin-top:4px}pre{white-space:pre-wrap;font-size:12px;color:var(--mut);margin:6px 0 0}
img{width:100%;border-radius:8px;margin-top:8px}table{width:100%;font-size:14px;border-collapse:collapse}td{padding:4px 2px;border-bottom:1px solid var(--line)}
td.n{text-align:right;font-variant-numeric:tabular-nums}.s{color:var(--ok);font-weight:700}.f{color:var(--bad);font-weight:700}
.sb{font-size:13px;padding:6px 9px;border-radius:8px;flex:none}</style></head><body>
<h1>Lacrosse scoop teach</h1>
<button id="go" onclick="act('start')">START</button>
<div class="row"><button id="ok" onclick="act('stop?outcome=success')">SUCCESS</button><button id="bad" onclick="act('stop?outcome=fail')">FAIL</button></div>
<div class="card"><div id="ph">-</div><div id="msg"></div><pre id="log"></pre></div>
<div class="card" id="res" style="display:none"><h2 id="rt"></h2><table id="tab"></table><img id="plot"><img id="photo"></div>
<div class="card"><h2 id="cnt">Sessions</h2><table id="list"></table></div>
<script>
let last=null;const $=i=>document.getElementById(i),f3=v=>(+v).toFixed(3);
async function act(a){$('go').disabled=$('ok').disabled=$('bad').disabled=true;try{await fetch('/'+a,{method:'POST'})}catch(e){}poll()}
async function lab(ts,v){await fetch('/label?ts='+ts+'&v='+v,{method:'POST'});list()}
async function list(){const L=await (await fetch('/sessions.json')).json();const ns=L.filter(x=>x.outcome=='success').length,nf=L.filter(x=>x.outcome=='fail').length;
 $('cnt').textContent='Sessions: '+L.length+'  (success '+ns+', fail '+nf+')';
 $('list').innerHTML=L.map(x=>'<tr><td>'+x.ts.slice(9)+'</td><td class="'+(x.outcome=='success'?'s':x.outcome=='fail'?'f':'')+'">'+x.outcome+'</td><td class="n">'+
  (x.ball?f3(x.ball[0])+' m':'')+'</td><td class="n">'+(x.gap>0?'<span class=f>lost '+(+x.gap).toFixed(1)+'s</span>':'')+'</td><td><button class="sb" style="background:var(--ok)" onclick="lab(\\''+x.ts+'\\',\\'success\\')">S</button> '+
  '<button class="sb" style="background:var(--bad)" onclick="lab(\\''+x.ts+'\\',\\'fail\\')">F</button></td></tr>').join('')}
async function poll(){try{const s=await (await fetch('/status')).json();
 $('ph').textContent=s.phase.toUpperCase()+(s.session?'  '+s.session:'');$('msg').textContent=s.msg;$('log').textContent=s.log.join('\\n');
 $('go').disabled=s.busy||s.phase!='idle';$('ok').disabled=$('bad').disabled=s.busy||s.phase!='recording';
 if(s.result&&s.result_session!==last){last=s.result_session;const r=s.result,b=r.ball&&(r.ball.ball_depth||r.ball.ball_size);
  $('rt').innerHTML=last+' <span class="'+(r.outcome=='success'?'s':'f')+'">'+r.outcome.toUpperCase()+'</span>';
  $('tab').innerHTML=[['ball (fwd / left) [m]',b?f3(b.forward)+' / '+f3(b.left):(r.ball&&r.ball.error)||'-'],['duration [s]',f3(r.duration_s)],
  ['tag detected',r.tag_detected+' / '+r.blackfly_frames+(r.tag_gap_total_s>0?' <b class=f>LOST '+(+r.tag_gap_total_s).toFixed(2)+' s</b>':'')],
  ['base forward range [m]',f3(r.base_fwd_range_m[0])+' .. '+f3(r.base_fwd_range_m[1])],['base |vx| max [m/s]',f3(r.base_vx_max_m_s)],
  ['base |ax| max [m/s2]',(+r.base_ax_max_m_s2).toFixed(1)],['shoulder [deg]',(+r.arm_q101_range_deg[0]).toFixed(1)+' .. '+(+r.arm_q101_range_deg[1]).toFixed(1)],
  ['elbow [deg]',(+r.arm_q102_range_deg[0]).toFixed(1)+' .. '+(+r.arm_q102_range_deg[1]).toFixed(1)]].map(x=>'<tr><td>'+x[0]+'</td><td class="n">'+x[1]+'</td></tr>').join('');
  $('plot').src='/sessions/'+last+'/plot.png?'+Date.now();$('photo').src='/sessions/'+last+'/ball_photo.jpg?'+Date.now();$('res').style.display='block';list()}
 }catch(e){$('msg').textContent='server not reachable'}}
setInterval(poll,1000);poll();list();
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/status":
            with lock:
                s = dict(state)
            if s.get("result"):
                s["result_session"] = s["session"]
            return self.send(200, json.dumps(s).encode(), "application/json")
        if path == "/sessions.json":
            return self.send(200, json.dumps(session_list()).encode(), "application/json")
        if path.startswith("/sessions/"):
            rel = os.path.normpath(path.lstrip("/"))
            fp = os.path.join(HERE, rel)
            if rel.startswith("sessions" + os.sep) and os.path.isfile(fp) and fp.endswith((".png", ".jpg")):
                return self.send(200, open(fp, "rb").read(), "image/png" if fp.endswith(".png") else "image/jpeg")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/label":
            ts, v = q.get("ts", [""])[0], q.get("v", [""])[0]
            f = os.path.join(SESS, os.path.basename(ts), "summary.json")
            if v in ("success", "fail") and os.path.isfile(f):
                s = json.load(open(f)); s["outcome"] = v; json.dump(s, open(f, "w"), indent=1)
                return self.send(200, b"ok", "text/plain")
            return self.send(400, b"bad label", "text/plain")
        outcome = q.get("outcome", [""])[0]
        with lock:
            busy, phase = state["busy"], state["phase"]
            ok = not busy and ((u.path == "/start" and phase == "idle") or
                               (u.path == "/stop" and phase == "recording" and outcome in ("success", "fail")))
            if ok:
                state["busy"] = True
                if u.path == "/start":
                    state.pop("result", None)
        if ok:
            target = (do_start,) if u.path == "/start" else (do_stop, outcome)
            threading.Thread(target=target[0], args=target[1:], daemon=True).start()
        self.send(200 if ok else 409, b"ok" if ok else b"busy", "text/plain")


if __name__ == "__main__":
    os.makedirs(SESS, exist_ok=True)
    print(f"teach web on http://0.0.0.0:{PORT}/", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
