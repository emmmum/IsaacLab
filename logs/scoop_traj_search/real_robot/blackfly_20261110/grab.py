# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Stream raw BayerRG8 frames + camera timestamp to stdout (system python3 / PySpin).
# Usage: grab.py SERIAL EXPOSURE_US   (EXPOSURE_US 0 = auto)
import PySpin, sys, struct
s = PySpin.System.GetInstance(); cams = s.GetCameras(); cam = cams.GetBySerial(sys.argv[1]); cam.Init()
exp = float(sys.argv[2])
if exp > 0:
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off); cam.ExposureTime.SetValue(exp)
else:
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
cam.AcquisitionFrameRateEnable.SetValue(False)
cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst)
cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual); cam.TLStream.StreamBufferCountManual.SetValue(100)
sys.stderr.write(f"exposure {cam.ExposureTime.GetValue():.0f} us, fps {cam.AcquisitionResultingFrameRate.GetValue():.1f}\n")
cam.BeginAcquisition(); out = sys.stdout.buffer
try:
    while True:
        im = cam.GetNextImage(2000)
        if not im.IsIncomplete():
            a = im.GetNDArray()
            out.write(struct.pack("<IIQQ", a.shape[1], a.shape[0], im.GetTimeStamp(), im.GetFrameID())); out.write(a.tobytes()); out.flush()
        im.Release()
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    cam.EndAcquisition(); cam.DeInit(); del cam; cams.Clear(); s.ReleaseInstance()
