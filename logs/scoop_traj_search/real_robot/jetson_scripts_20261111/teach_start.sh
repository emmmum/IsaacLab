#!/bin/bash
# Teach recording: arm (grav_comp_rec CSV), chassis odometry, AprilTag. Stop with teach_stop.sh.
cd ~/robomaster_test; TS=$(date +%Y%m%d_%H%M%S); echo $TS > TEACH_TS; rm -f TEACH_STOP
# hand the arm over: stop grav_comp (latches Kp 20/50 at current q), then start the recording version
if pgrep -f "^python3 grav_comp" >/dev/null; then
  touch STOP_HOLD; for i in $(seq 50); do pgrep -f "^python3 grav_comp" >/dev/null || break; sleep 0.2; done
fi
pgrep -f "^python3 grav_comp" && { echo "grav_comp still running, abort"; exit 1; }
Q=$(python3 param_read.py | awk '/^10[12]:/{print $3}' | tr '\n' ' ')
echo "handover q: $Q"
setsid nohup python3 grav_comp_rec.py --init $Q --kp-from 20 50 ${NOWALL:+--no-wall} --csv teach_arm_$TS.csv > teach_arm_$TS.out 2>&1 < /dev/null &
[ -z "$NOODOM" ] && setsid nohup python3 odom_rec.py teach_odom_$TS.csv > teach_odom_$TS.out 2>&1 < /dev/null &
TAG_RES=640 setsid nohup python3 tag_track.py teach_tag_$TS.csv 1800 TEACH_STOP > teach_tag_$TS.out 2>&1 < /dev/null &
sleep 5; tail -2 teach_arm_$TS.out; wc -l teach_tag_$TS.csv; echo "TEACH $TS recording"
