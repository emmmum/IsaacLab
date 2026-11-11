#!/bin/sh
# ssh to the Jetson over a persistent master connection (password via askpass)
D=$(dirname "$(readlink -f "$0")")
SSH_ASKPASS=$D/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=x setsid ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no \
  -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=4h -o ConnectTimeout=8 jetson@10.155.35.156 "$@"
