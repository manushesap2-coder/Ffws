"""Serhat grade: post-LUT Rec.709 float RGB -> finished look."""
import numpy as np, cv2, sys
sys.path.insert(0, '/tmp/jobs/serhat/lib')
from core import luma, smooth_curve

def grade_s(x, p):
    x = np.clip(x, 0, 1.2)
    ev = p.get('ev', 0.0)
    if ev:
        x = np.clip(x, 0, None) ** 2.2 * (2.0 ** ev)
        x = x ** (1 / 2.2)
    # shadow lift (toe), weighted to dark areas only
    sl = p.get('shadows', 0.0)
    if sl:
        Y = luma(x)[..., None]
        w = np.exp(-np.clip(Y, 0, None) / 0.22)
        x = x + sl * w * (0.55 - np.minimum(Y, 0.55)) * (x / np.maximum(Y, 1e-3)) ** 0.5
    # highlight roll-off (soft shoulder from 0.72)
    hl = p.get('hl', 0.0)
    if hl > 0:
        d = np.maximum(x - 0.72, 0.0)
        x = np.where(x > 0.72, 0.72 + d / (1 + hl * d / 0.28), x)
    x = np.clip(x, 0, 1)
    x = smooth_curve(x, p.get('contrast', 1.12), p.get('pivot', 0.42))
    Y = luma(x)[..., None]; c = x - Y
    chroma = np.sqrt((c ** 2).sum(-1, keepdims=True))
    sat, vib = p.get('sat', 1.1), p.get('vib', 0.3)
    k = sat * (1 + vib * np.clip(0.2 - chroma, 0, 0.2) / 0.2) / (1 + vib)
    x = Y + c * k
    gain = np.array(p.get('gain', [1.02, 1.0, 0.975]), np.float32)
    lift = np.array(p.get('lift', [0.0, 0.004, 0.012]), np.float32)
    x = x * gain + lift * (1 - x)
    sky = p.get('sky', 0.0)
    if sky > 0:
        h = x.shape[0]; ext = p.get('sky_ext', 0.3)
        Y2 = luma(x)
        ramp = np.clip(1.0 - np.arange(h, dtype=np.float32) / (h * ext), 0, 1)[:, None]
        cc = np.sqrt(((x - Y2[..., None]) ** 2).sum(-1))
        m = np.clip((Y2 - 0.60) / 0.22, 0, 1) * np.clip(1 - cc / 0.16, 0, 1) * ramp
        m = cv2.GaussianBlur(m, (0, 0), 8)[..., None] * sky
        top = np.array([0.36, 0.56, 0.86], np.float32); bot = np.array([0.70, 0.80, 0.93], np.float32)
        g = top + (bot - top) * (np.arange(h, dtype=np.float32) / (h * ext)).clip(0, 1)[:, None, None]
        x = x * (1 - m) + (x * g * 1.12) * m
    x = np.clip(x, 0, 1)
    cl = p.get('clarity', 0.15)
    if cl > 0:
        Y = luma(x); small = cv2.resize(Y, (Y.shape[1] // 4, Y.shape[0] // 4), interpolation=cv2.INTER_AREA)
        blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        x = x + (cl * (Y - blur))[..., None]
    sh = p.get('sharpen', 0.3)
    if sh > 0:
        b = cv2.GaussianBlur(x, (0, 0), 1.0)
        x = x + sh * (x - b)
    return np.clip(x, 0, 1)
