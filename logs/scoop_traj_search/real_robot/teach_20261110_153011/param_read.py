# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only: CyberGear params via type-17 frames (does NOT enable/disable or move)."""
import math, struct, time
import can
bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
def param(mid, idx, fmt):
    while bus.recv(timeout=0.005): pass
    bus.send(can.Message(arbitration_id=(17 << 24) | (254 << 8) | mid, data=list(struct.pack("<HH", idx, 0)) + [0]*4, is_extended_id=True))
    t0 = time.time()
    while time.time() - t0 < 0.3:
        r = bus.recv(timeout=0.05)
        if r and (r.arbitration_id >> 24) == 17 and ((r.arbitration_id >> 8) & 0xFF) == mid:
            return struct.unpack("<" + fmt, bytes(r.data[4:4 + struct.calcsize(fmt)]))[0]
for mid in (101, 102):
    mp = param(mid, 0x7019, "f"); rot = param(mid, 0x701D, "h"); mode = param(mid, 0x7005, "B"); vb = param(mid, 0x701C, "f"); iq = param(mid, 0x701A, "f")
    if mp is None: print(f"{mid}: no reply"); continue
    print(f"{mid}: mechPos {math.degrees(mp):+8.2f} deg | rotation {rot} | run_mode {mode} | iqf {iq:+.3f} A | VBUS {vb:.2f} V")
bus.shutdown()
