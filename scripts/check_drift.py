"""Drift check: L/S trajectory + contact strip at flagged timesteps.
Usage: python check_drift.py output/nb_iter1.mp4 [sheet_out.png]
"""
import sys
import cv2
import numpy as np

vid = sys.argv[1]
sheet_out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/drift_sheet.png"
cap = cv2.VideoCapture(vid)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{vid}: {int(cap.get(3))}x{int(cap.get(4))} {n} frames")
print("t(s)   Lmean   Smean")
for t in range(0, 185, 15):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    print(f"{t:4d} {lab[:, :, 0].mean():7.1f} {hsv[:, :, 1].mean():7.1f}")
thumbs = []
for t in [10, 75, 150]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    thumbs.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (480, 270)))
cap.release()
cv2.imwrite(sheet_out, cv2.cvtColor(np.hstack(thumbs), cv2.COLOR_RGB2BGR))
print("sheet:", sheet_out)
