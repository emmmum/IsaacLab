# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-cut the arm slice (full, flushed log) and re-run process() for existing sessions. Usage: reprocess.py ARM_LIVE_FILE TS..."""
import json, os, sys
import numpy as np
sys.argv, args = [sys.argv[0]], sys.argv[1:]
import teach_web as tw
arm_file = args[0]
for ts in args[1:]:
    d = os.path.join(tw.SESS, ts)
    try:
        old = json.load(open(os.path.join(d, "summary.json"))) if os.path.isfile(os.path.join(d, "summary.json")) else {}
        a = np.genfromtxt(os.path.join(d, "blackfly.csv"), delimiter=",", names=True, invalid_raise=False)
        tcap = a["t_cam_s"] + np.min(a["host_t"] - a["t_cam_s"])
        t_start, t_stop = old.get("t_start", tcap[0]), old.get("t_stop", tcap[-1])
        off = float(np.mean(old.get("clock_offset_s", [0.047])))
        tw.fetch_arm(d, arm_file, t_start + off - 0.5, t_stop + off + 0.5)
        s = tw.process(d, t_start, t_stop, off)
        s.update(outcome=old.get("outcome", "?"), t_start=t_start, t_stop=t_stop, clock_offset_s=old.get("clock_offset_s", [0.047]),
                 ball=old.get("ball") or (json.load(open(os.path.join(d, "ball_floor.json"))) if os.path.isfile(os.path.join(d, "ball_floor.json")) else {}),
                 reprocessed="full arm log")
        json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=1)
        print(ts, "ok  dur %.1f  q101 %s" % (s["duration_s"], np.round(s["arm_q101_range_deg"], 1)), flush=True)
    except Exception as e:
        print(ts, "FAILED", e, flush=True)
