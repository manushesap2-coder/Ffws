"""Frame IO, grading and small image helpers shared by the compositor."""
import subprocess, numpy as np, cv2

class Reader:
    """Sequential 16-bit RGB frame reader for an intermediate clip (supports forward seeking)."""
    def __init__(self, path, w, h, fps=60000/1001):
        self.path, self.w, self.h, self.fps = path, w, h, fps
        self.proc = None; self.pos = -1; self.cur = None
        self._open(0)

    def _open(self, idx):
        if self.proc: self.proc.kill()
        t = idx / self.fps
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t:.5f}', '-i', self.path, '-f', 'rawvideo',
                                      '-pix_fmt', 'rgb48le', '-'], stdout=subprocess.PIPE, bufsize=10**8)
        self.pos = idx - 1; self.cur = None

    def get(self, idx):
        idx = max(0, int(idx))
        if idx < self.pos or idx > self.pos + 90:
            self._open(idx)
        n = self.w * self.h * 3 * 2
        while self.pos < idx:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:        # past the end: hold last frame
                return self.cur
            self.pos += 1
            if self.pos == idx:
                self.cur = np.frombuffer(buf, np.uint16).reshape(self.h, self.w, 3).astype(np.float32) / 65535.0
        return self.cur

    def close(self):
        if self.proc: self.proc.kill()


def luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722

def smooth_curve(x, contrast=1.0, pivot=0.45):
    """Filmic S-curve around pivot; contrast>1 increases mid contrast, soft shoulder/toe."""
    x = np.clip(x, 0, 1)
    lo = pivot * (x / pivot) ** contrast
    hi = 1 - (1 - pivot) * ((1 - x) / (1 - pivot)) ** contrast
    return np.where(x < pivot, lo, hi)

def grade(img, p):
    """img float32 RGB [0,1]. p: dict of grade params."""
    x = (img - p.get('black', 0.0)) / max(1e-3, p.get('white', 1.0) - p.get('black', 0.0))
    x = np.clip(x, 0, 1.2)
    # highlight roll-off before the curve (keeps clipped drone skies from going flat white)
    hl = p.get('hl_rolloff', 0.0)
    if hl > 0:
        d = np.maximum(x - 0.75, 0.0)
        x = np.where(x > 0.75, 0.75 + d / (1 + hl * d / 0.25), x)
    x = np.clip(x, 0, 1)
    if p.get('gamma', 1.0) != 1.0:
        x = x ** (1.0 / p['gamma'])
    x = smooth_curve(x, p.get('contrast', 1.25), p.get('pivot', 0.45))
    # saturation with vibrance-style protection of already saturated pixels
    Y = luma(x)[..., None]
    c = x - Y
    chroma = np.sqrt((c ** 2).sum(-1, keepdims=True))
    sat = p.get('sat', 1.4)
    vib = p.get('vib', 0.35)
    k = sat * (1 + vib * np.clip(0.25 - chroma, 0, 0.25) / 0.25) / (1 + vib)
    x = Y + c * k
    # split toning: warm highlights / cool shadows (per-channel lift & gain)
    lift = np.array(p.get('lift', [0.0, 0.0, 0.0]), np.float32)
    gain = np.array(p.get('gain', [1.0, 1.0, 1.0]), np.float32)
    x = x * gain + lift * (1 - x)
    # sky: deepen bright, low-saturation pixels in the upper frame toward a blue gradient
    sky = p.get('sky', 0.0)
    if sky > 0:
        h = x.shape[0]
        Y2 = luma(x)
        ramp = np.clip(1.0 - np.arange(h, dtype=np.float32) / (h * p.get('sky_extent', 0.35)), 0, 1)[:, None]
        cc = np.sqrt(((x - Y2[..., None]) ** 2).sum(-1))
        m = np.clip((Y2 - 0.62) / 0.25, 0, 1) * np.clip(1 - cc / 0.18, 0, 1) * ramp
        m = cv2.GaussianBlur(m, (0, 0), 6)[..., None] * sky
        top = np.array([0.30, 0.52, 0.86], np.float32); bot = np.array([0.62, 0.78, 0.96], np.float32)
        grad = top + (bot - top) * (np.arange(h, dtype=np.float32) / (h * p.get('sky_extent', 0.35))).clip(0, 1)[:, None, None]
        x = x * (1 - m) + (x * grad * 1.08) * m
    x = np.clip(x, 0, 1)
    # clarity (large radius local contrast on luma) and fine sharpening
    cl = p.get('clarity', 0.3)
    if cl > 0:
        Y = luma(x)
        blur = cv2.GaussianBlur(Y, (0, 0), 18)
        x = x + (cl * (Y - blur))[..., None]
    sh = p.get('sharpen', 0.35)
    if sh > 0:
        b = cv2.GaussianBlur(x, (0, 0), 1.1)
        x = x + sh * (x - b)
    return np.clip(x, 0, 1)

def auto_levels(img, lo=0.4, hi=99.6):
    Y = luma(img)
    return float(np.percentile(Y, lo)), float(np.percentile(Y, hi))
