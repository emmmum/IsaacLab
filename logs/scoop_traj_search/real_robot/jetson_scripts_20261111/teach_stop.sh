#!/bin/bash
# Stop the odometry + tag loggers (arm stays in gravity comp and keeps logging until STOP_HOLD).
cd ~/robomaster_test; TS=$(cat TEACH_TS); touch TEACH_STOP; pkill -TERM -f "odom_rec.py teach_odom_$TS" ; sleep 2
echo "stopped $(date +%T)" >> teach_arm_$TS.out
wc -l teach_arm_$TS.csv teach_odom_$TS.csv teach_tag_$TS.csv; tail -1 teach_tag_$TS.out
