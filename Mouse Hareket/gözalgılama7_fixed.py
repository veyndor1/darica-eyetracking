import collections
import ctypes
import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np

# --- Landmark indeksleri ---
LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Göz köşe / üst-alt noktaları
L_OUTER, L_INNER, L_TOP, L_BOT = 33, 133, 159, 145
R_OUTER, R_INNER, R_TOP, R_BOT = 263, 362, 386, 374

PROCESS_WIDTH = 640
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

user32   = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

# Kalibrasyon: 13 nokta
CAL_POINTS_REL = [
    (0.50, 0.50),
    (0.08, 0.08), (0.92, 0.08),
    (0.08, 0.92), (0.92, 0.92),
    (0.50, 0.08), (0.50, 0.92),
    (0.08, 0.50), (0.92, 0.50),
    (0.29, 0.29), (0.71, 0.29),
    (0.29, 0.71), (0.71, 0.71),
]
CAL_WAIT    = 1.2
CAL_COLLECT = 1.8
CAL_RIDGE   = 0.02
EAR_BLINK   = 0.40

DEAD_ZONE            = 0
DOUBLE_BLINK_MAX_GAP = 1.5
GAZE_X_OFFSET        = 0
GAZE_Y_OFFSET        = -2
GAZE_Y_FLIP          = 1

# ── FIX 4: 0.12 → 0.22 rad ─────────────────────────────────────────────────
# 0.12 rad (~7°) çok düşüktü; normal kafa mikro-hareketleri sürekli tetikliyordu.
# 0.22 rad (~13°) yalnızca gerçek sarsıntılarda imleci dondurur.
HEAD_DELTA_GATE = 0.22

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

VEL_DAMP_THRESHOLD   = 0.6
ADAPTIVE_DEAD_BASE   = DEAD_ZONE
ADAPTIVE_DEAD_BONUS  = 5
ADAPTIVE_DEAD_SPEED  = 12.0

# TPS için iris ve kafa açısı harmanlama ağırlığı
# iris_rel ≈ ±0.12, yaw/pitch ≈ ±0.30 → 1.5× iris, katkıları dengeler
TPS_IRIS_WEIGHT = 1.5


# ─────────────────────────────────────────────
# One Euro Filter
# ─────────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, min_cutoff=0.9, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x         = None
        self._dx        = 0.0
        self._t         = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self._x is None:
            self._x, self._t = float(x), t
            return self._x
        dt       = max(t - self._t, 1e-6)
        dx       = (x - self._x) / dt
        a_d      = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff   = self.min_cutoff + self.beta * abs(self._dx)
        a        = self._alpha(cutoff, dt)
        self._x  = a * float(x) + (1.0 - a) * self._x
        self._t  = t
        return self._x

    def reset(self):
        self._x  = None
        self._dx = 0.0
        self._t  = None


# ─────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────

def lm_to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def eye_bbox(points, pad_x=0.25, pad_y=0.40):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pw = int((x1 - x0) * pad_x)
    ph = int((y1 - y0) * pad_y)
    return x0 - pw, y0 - ph, x1 + pw, y1 + ph


def draw_iris(frame, iris_px):
    cx, cy = iris_px[0]
    if len(iris_px) > 1:
        radius = int(np.mean([np.hypot(p[0]-cx, p[1]-cy) for p in iris_px[1:]]))
    else:
        radius = 8
    radius = max(radius, 2)
    cv2.circle(frame, (cx, cy), radius, (0, 0, 220), 2)
    cv2.circle(frame, (cx, cy), 2, (255, 230, 0), -1)


def compute_ear(lms):
    def _ear(h1, h2, v1, v2, v3, v4):
        horiz = math.hypot(lms[h1].x - lms[h2].x, lms[h1].y - lms[h2].y)
        if horiz < 1e-6:
            return 1.0
        vert1 = math.hypot(lms[v1].x - lms[v2].x, lms[v1].y - lms[v2].y)
        vert2 = math.hypot(lms[v3].x - lms[v4].x, lms[v3].y - lms[v4].y)
        return (vert1 + vert2) / (2.0 * horiz)
    ear_l = _ear(33, 133, 159, 145, 158, 153)
    ear_r = _ear(263, 362, 386, 374, 385, 380)
    return (ear_l + ear_r) / 2.0


def get_head_angles(transform_matrix_data):
    mat   = np.array(transform_matrix_data.data, dtype=float).reshape(4, 4)
    R     = mat[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    pitch = float(rvec[0, 0])
    yaw   = float(rvec[1, 0])
    return yaw, pitch


def compute_gaze_features(lms, transform_matrix_data):
    yaw, pitch = get_head_angles(transform_matrix_data)

    def _iris_mean(indices):
        n = len(indices)
        return (
            sum(lms[i].x for i in indices) / n,
            sum(lms[i].y for i in indices) / n,
        )

    def _eye_span(indices):
        xs = [lms[i].x for i in indices]
        ys = [lms[i].y for i in indices]
        return min(xs), max(xs), min(ys), max(ys)

    l_xmin, l_xmax, l_ymin, l_ymax = _eye_span(LEFT_EYE)
    l_ix, l_iy = _iris_mean(LEFT_IRIS)
    l_ew = l_xmax - l_xmin
    l_eh = l_ymax - l_ymin

    r_xmin, r_xmax, r_ymin, r_ymax = _eye_span(RIGHT_EYE)
    r_ix, r_iy = _iris_mean(RIGHT_IRIS)
    r_ew = r_xmax - r_xmin
    r_eh = r_ymax - r_ymin

    left_rx  = (l_ix - (l_xmin + l_xmax) / 2.0) / l_ew if l_ew > 1e-5 else None
    right_rx = (r_ix - (r_xmin + r_xmax) / 2.0) / r_ew if r_ew > 1e-5 else None

    if left_rx is not None and right_rx is not None:
        iris_rel_x = (left_rx + right_rx) / 2.0
    elif left_rx is not None:
        iris_rel_x = left_rx
    elif right_rx is not None:
        iris_rel_x = right_rx
    else:
        return None

    left_ry  = (l_iy - (l_ymin + l_ymax) / 2.0) / l_eh if l_eh > 1e-5 else None
    right_ry = (r_iy - (r_ymin + r_ymax) / 2.0) / r_eh if r_eh > 1e-5 else None

    if left_ry is not None and right_ry is not None:
        iris_rel_y = (left_ry + right_ry) / 2.0
    elif left_ry is not None:
        iris_rel_y = left_ry
    elif right_ry is not None:
        iris_rel_y = right_ry
    else:
        return None

    vergence = (left_rx - right_rx) if (left_rx is not None and right_rx is not None) else 0.0
    return [yaw, pitch, iris_rel_x, iris_rel_y * GAZE_Y_FLIP, vergence]


def process_frame(cap, face_landmarker):
    ret, frame = cap.read()
    if not ret:
        return None, None, 0, 0
    orig_h, orig_w = frame.shape[:2]
    scale = PROCESS_WIDTH / orig_w if orig_w > PROCESS_WIDTH else 1.0
    small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale))) if scale < 1.0 else frame
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_img)
    return frame, result, orig_w, orig_h


# ─────────────────────────────────────────────
# Kalman Filtresi (2D konum + hız)
# ─────────────────────────────────────────────

class KalmanGaze:
    # ── FIX 2: process_noise 55→20, measure_noise 5500→600 ─────────────────
    # Eski değerle Kalman ölçümleri neredeyse hiç dinlemiyordu (R=5500 ≈ ±74px hata
    # varsayımı). 600 ile ölçümler çok daha etkili olur; cursor doğru yere gider.
    def __init__(self, process_noise=20.0, measure_noise=600.0):
        dt = 1.0 / 30.0
        self.x = np.array([[SCREEN_W / 2.0],
                            [SCREEN_H / 2.0],
                            [0.0],
                            [0.0]])
        self.P = np.eye(4) * 2000.0
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measure_noise

    def update(self, meas):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.array([[float(meas[0])], [float(meas[1])]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self._damp_velocity()
        return self._clamp()

    def predict_only(self):
        # ── FIX 1: Hızı ANINDA sıfırla — en kritik düzeltme ───────────────
        # Eski kod hızı taşıyordu: yüz kaybedilince cursor son hızıyla
        # ekranın köşesine fırlatılıp y=0'da (ekran üstü) kilitleniyordu.
        # Hızı önce sıfırlayarak tahmin yapıyoruz → cursor yerinde donuyor.
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._damp_velocity()
        return self._clamp()

    def _damp_velocity(self):
        if abs(self.x[2, 0]) < VEL_DAMP_THRESHOLD:
            self.x[2, 0] = 0.0
        if abs(self.x[3, 0]) < VEL_DAMP_THRESHOLD:
            self.x[3, 0] = 0.0

    def _clamp(self):
        return (
            int(np.clip(self.x[0, 0], 0, SCREEN_W - 1)),
            int(np.clip(self.x[1, 0], 0, SCREEN_H - 1)),
        )


# ─────────────────────────────────────────────
# Polinom özellik fonksiyonları — X ve Y tamamen ayrı
# ─────────────────────────────────────────────

def _feat_x(yaw, iris_rel_x):
    return np.array([
        yaw,
        iris_rel_x,
        yaw * iris_rel_x,
        yaw ** 2,
        iris_rel_x ** 2,
        1.0
    ], dtype=float)


def _feat_y(pitch, iris_rel_y):
    return np.array([
        pitch,
        iris_rel_y,
        pitch * iris_rel_y,
        pitch ** 2,
        iris_rel_y ** 2,
        1.0
    ], dtype=float)


def _ridge_solve(A, b, lam):
    M = A.T @ A
    M += lam * np.eye(M.shape[0])
    return np.linalg.solve(M, A.T @ b)


# ─────────────────────────────────────────────
# Thin Plate Spline
# ─────────────────────────────────────────────

class ThinPlateSpline:
    """
    2D → 2D Thin Plate Spline.
    Kalibrasyon noktalarında hata ≈ 0, arada C² pürüzsüz interpolasyon.
    Polynomial ridge regression'dan çok daha hassas — 13 noktanın hepsine tam oturur.
    """

    @staticmethod
    def _rbf(r: float) -> float:
        return r * r * math.log(r + 1e-10)

    def fit(self, src_pts, dst_x, dst_y, lam: float = 1e-4):
        self._src = np.array(src_pts, dtype=float)
        N = len(self._src)

        K = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                r = math.hypot(self._src[i, 0] - self._src[j, 0],
                               self._src[i, 1] - self._src[j, 1])
                K[i, j] = K[j, i] = self._rbf(r)

        P = np.hstack([np.ones((N, 1), dtype=float), self._src])
        Z = np.zeros((3, 3), dtype=float)
        M = np.block([
            [K + lam * np.eye(N), P],
            [P.T,                 Z]
        ])

        rhs_x = np.concatenate([np.array(dst_x, dtype=float), [0.0, 0.0, 0.0]])
        rhs_y = np.concatenate([np.array(dst_y, dtype=float), [0.0, 0.0, 0.0]])
        sol = np.linalg.solve(M, np.column_stack([rhs_x, rhs_y]))

        self._w = sol[:N]
        self._a = sol[N:]

    def predict(self, gx: float, gy: float):
        rbf_vals = np.array([
            self._rbf(math.hypot(gx - s[0], gy - s[1]))
            for s in self._src
        ])
        out = rbf_vals @ self._w + self._a[0] + self._a[1] * gx + self._a[2] * gy
        return float(out[0]), float(out[1])


# ─────────────────────────────────────────────
# GazeMapper — kalibrasyon + mapping
# ─────────────────────────────────────────────

class GazeMapper:
    def __init__(self):
        self.Mx        = None
        self.My        = None
        self.tps       = None          # FIX 3: TPS modeli
        self.kalman    = KalmanGaze()
        self.sent_pos  = (SCREEN_W // 2, SCREEN_H // 2)
        self._prev_yaw   = None
        self._prev_pitch = None
        self._oef_iris_x   = OneEuroFilter(min_cutoff=0.9, beta=0.04)
        self._oef_iris_y   = OneEuroFilter(min_cutoff=0.9, beta=0.04)
        self._oef_yaw      = OneEuroFilter(min_cutoff=0.6, beta=0.03)
        self._oef_pitch    = OneEuroFilter(min_cutoff=0.6, beta=0.03)
        self._oef_vergence = OneEuroFilter(min_cutoff=0.9, beta=0.04)
        self._nx_a = 1.0;  self._nx_b = 0.0
        self._ny_a = 1.0;  self._ny_b = 0.0

    def calibrate(self, features_list, screen_pts):
        # ── Polinom Ridge Regresyon (yedek model olarak tutulur) ────────────
        Ax = np.array([_feat_x(f[0], f[2]) for f in features_list], dtype=float)
        Ay = np.array([_feat_y(f[1], f[3]) for f in features_list], dtype=float)
        bx = np.array([p[0] for p in screen_pts], dtype=float)
        by = np.array([p[1] for p in screen_pts], dtype=float)
        self.Mx = _ridge_solve(Ax, bx, CAL_RIDGE)
        self.My = _ridge_solve(Ay, by, CAL_RIDGE)

        # ── FIX 5: Normalizasyon cap 3.0 → 5.0 ────────────────────────────
        # 3.0 çok kısıtlayıcıydı; geniş kalibrasyon aralığında ekranın
        # kenarlarına ulaşılamıyordu.
        pred_x = Ax @ self.Mx
        pred_y = Ay @ self.My
        px_lo, px_hi = pred_x.min(), pred_x.max()
        py_lo, py_hi = pred_y.min(), pred_y.max()
        if px_hi > px_lo:
            self._nx_a = min(SCREEN_W / (px_hi - px_lo), 5.0)
            self._nx_b = -self._nx_a * px_lo
        else:
            self._nx_a, self._nx_b = 1.0, 0.0
        if py_hi > py_lo:
            self._ny_a = min(SCREEN_H / (py_hi - py_lo), 5.0)
            self._ny_b = -self._ny_a * py_lo
        else:
            self._ny_a, self._ny_b = 1.0, 0.0

        # ── FIX 3: TPS fit ─────────────────────────────────────────────────
        # 2D gaze uzayı: (iris_x * ağırlık + yaw, iris_y * ağırlık + pitch)
        # TPS, kalibrasyon noktalarında sıfır hata verir; arada pürüzsüz
        # interpolasyon yapar. Polynomial'dan çok daha hassas.
        try:
            gaze_2d = [
                (f[2] * TPS_IRIS_WEIGHT + f[0],   # iris_rel_x * w + yaw
                 f[3] * TPS_IRIS_WEIGHT + f[1])   # iris_rel_y * w + pitch
                for f in features_list
            ]
            self.tps = ThinPlateSpline()
            self.tps.fit(
                gaze_2d,
                [p[0] for p in screen_pts],
                [p[1] for p in screen_pts],
                lam=1e-3,
            )
        except Exception as e:
            # TPS çözüm başarısız olursa polynomial'a geri dön
            print(f"[UYARI] TPS fit başarısız, polynomial kullanılacak: {e}")
            self.tps = None

        # Kalman ve filtreleri sıfırla
        self.kalman      = KalmanGaze()
        self.sent_pos    = (SCREEN_W // 2, SCREEN_H // 2)
        self._prev_yaw   = None
        self._prev_pitch = None
        for f in (self._oef_iris_x, self._oef_iris_y,
                  self._oef_yaw, self._oef_pitch, self._oef_vergence):
            f.reset()

    def map(self, features, blinking=False):
        if self.Mx is None or features is None:
            # Yüz yok → cursor son konumda dondur (FIX 1 zaten hızı sıfırlıyor)
            return self.kalman.predict_only()

        # ── FIX 5 (kırpma): Pozisyonu dondur, tahmin yapma ─────────────────
        # Eski kod: blinking → predict_only() → cursor hızla kayıyor.
        # Düzeltme: kırpma sırasında son bilinen konuma dön.
        if blinking:
            return self.sent_pos

        yaw, pitch, iris_rel_x, iris_rel_y, vergence = features

        # ── FIX 4: HEAD_DELTA_GATE kontrolü (artık 0.22 rad) ───────────────
        if self._prev_yaw is not None:
            if (abs(yaw - self._prev_yaw) > HEAD_DELTA_GATE or
                    abs(pitch - self._prev_pitch) > HEAD_DELTA_GATE):
                self._prev_yaw, self._prev_pitch = yaw, pitch
                # Ani kafa hareketi → son konumda dondur (predict_only değil)
                return self.sent_pos
        self._prev_yaw, self._prev_pitch = yaw, pitch

        # ── One Euro filtreleme ──────────────────────────────────────────────
        t = time.perf_counter()
        yaw        = self._oef_yaw.update(yaw, t)
        pitch      = self._oef_pitch.update(pitch, t)
        iris_rel_x = self._oef_iris_x.update(iris_rel_x, t)
        iris_rel_y = self._oef_iris_y.update(iris_rel_y, t)
        vergence   = self._oef_vergence.update(vergence, t)

        # ── FIX 3: TPS varsa TPS kullan, yoksa polynomial'a geri dön ────────
        if self.tps is not None:
            gx = iris_rel_x * TPS_IRIS_WEIGHT + yaw
            gy = iris_rel_y * TPS_IRIS_WEIGHT + pitch
            raw_x, raw_y = self.tps.predict(gx, gy)
            sx = float(np.clip(raw_x + GAZE_X_OFFSET, 0, SCREEN_W - 1))
            sy = float(np.clip(raw_y + GAZE_Y_OFFSET, 0, SCREEN_H - 1))
        else:
            raw_x = float(_feat_x(yaw, iris_rel_x) @ self.Mx)
            raw_y = float(_feat_y(pitch, iris_rel_y) @ self.My)
            sx = float(np.clip(self._nx_a * raw_x + self._nx_b + GAZE_X_OFFSET, 0, SCREEN_W - 1))
            sy = float(np.clip(self._ny_a * raw_y + self._ny_b + GAZE_Y_OFFSET, 0, SCREEN_H - 1))

        smoothed = self.kalman.update((sx, sy))

        # ── Adaptif ölü bölge ────────────────────────────────────────────────
        speed = math.hypot(self.kalman.x[2, 0], self.kalman.x[3, 0])
        scale = max(0.0, 1.0 - speed / ADAPTIVE_DEAD_SPEED)
        dead  = ADAPTIVE_DEAD_BASE + int(ADAPTIVE_DEAD_BONUS * scale)

        dx = abs(smoothed[0] - self.sent_pos[0])
        dy = abs(smoothed[1] - self.sent_pos[1])
        if dx < dead and dy < dead:
            self.kalman.x[0, 0] = float(self.sent_pos[0])
            self.kalman.x[1, 0] = float(self.sent_pos[1])
            self.kalman.x[2, 0] = 0.0
            self.kalman.x[3, 0] = 0.0
            return self.sent_pos
        self.sent_pos = smoothed
        return smoothed


# ─────────────────────────────────────────────
# Çift kırpma tespiti
# ─────────────────────────────────────────────

class BlinkDetector:
    def __init__(self):
        self.blink_frames  = 0
        self.in_blink      = False
        self.blink_times   = collections.deque(maxlen=3)

    def update(self, ear, now):
        double_blink = False

        if ear < EAR_BLINK:
            self.blink_frames += 1
            is_blinking = self.blink_frames >= 2
        else:
            if self.in_blink and self.blink_frames >= 2:
                self.blink_times.append(now)
                if (len(self.blink_times) >= 2
                        and now - self.blink_times[-2] <= DOUBLE_BLINK_MAX_GAP):
                    double_blink = True
                    self.blink_times.clear()
            self.blink_frames = 0
            is_blinking = False

        self.in_blink = is_blinking
        return is_blinking, double_blink


# ─────────────────────────────────────────────
# Kalibrasyon ekranı
# ─────────────────────────────────────────────

def _trim_mean(samples, trim=0.15):
    arr = np.array(samples, dtype=float)
    n   = len(arr)
    lo  = int(n * trim)
    hi  = n - lo
    result = []
    for col in range(arr.shape[1]):
        s = np.sort(arr[:, col])
        result.append(np.mean(s[lo:hi]) if hi > lo else np.mean(s))
    return result


def _best_samples(samples):
    cutoff = int(len(samples) * 0.45)
    late   = samples[cutoff:] if len(samples) > 12 else samples

    if len(late) < 4:
        return late
    arr = np.array(late, dtype=float)
    vel = np.sqrt(np.diff(arr[:, 2]) ** 2 + np.diff(arr[:, 3]) ** 2)
    thresh = np.percentile(vel, 60)
    keep   = [0] + [i + 1 for i, v in enumerate(vel) if v <= thresh]
    stable = [late[i] for i in keep]
    return stable if len(stable) >= 5 else late


def run_calibration(cap, face_landmarker):
    cal_win = "KALIBRASYON"
    cv2.namedWindow(cal_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(cal_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    features_list = []
    screen_pts    = []
    pt_idx = 0

    while pt_idx < len(CAL_POINTS_REL):
        rx, ry = CAL_POINTS_REL[pt_idx]
        sx = int(rx * SCREEN_W)
        sy = int(ry * SCREEN_H)

        pt_start = time.perf_counter()
        samples  = []

        while True:
            frame, result, _, _ = process_frame(cap, face_landmarker)
            if frame is None:
                cv2.destroyWindow(cal_win)
                return None

            elapsed = time.perf_counter() - pt_start

            if (result.face_landmarks
                    and result.facial_transformation_matrixes
                    and elapsed > CAL_WAIT):
                lms = result.face_landmarks[0]
                ear = compute_ear(lms)
                if ear > EAR_BLINK:
                    feats = compute_gaze_features(lms, result.facial_transformation_matrixes[0])
                    if feats:
                        samples.append(feats)

            cal_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

            # Yüz tespiti göstergesi
            face_ok = bool(result.face_landmarks and result.facial_transformation_matrixes)
            indicator_color = (0, 200, 0) if face_ok else (0, 0, 220)
            indicator_text  = "YUZ ALGILANDI" if face_ok else "YUZ ALGILANAMADI — KAMERAYA BAKIN"
            cv2.putText(cal_frame, indicator_text, (30, SCREEN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2, cv2.LINE_AA)

            msg = f"Noktaya bakin  ({pt_idx + 1} / {len(CAL_POINTS_REL)})"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(cal_frame, msg, ((SCREEN_W - tw) // 2, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(cal_frame, "q = iptal", (30, SCREEN_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1, cv2.LINE_AA)

            for i, (frx, fry) in enumerate(CAL_POINTS_REL):
                if i != pt_idx:
                    fx, fy = int(frx * SCREEN_W), int(fry * SCREEN_H)
                    cv2.circle(cal_frame, (fx, fy), 8, (40, 40, 40), -1)

            blink_anim = int(elapsed * 6) % 2 == 0
            dot_color  = (0, 255, 255) if blink_anim else (0, 120, 120)
            if elapsed > CAL_WAIT:
                arc_pct   = min((elapsed - CAL_WAIT) / CAL_COLLECT, 1.0)
                arc_angle = int(360 * arc_pct)
                cv2.ellipse(cal_frame, (sx, sy), (34, 34), -90, 0, arc_angle,
                            (0, 230, 0), 4, cv2.LINE_AA)
            cv2.circle(cal_frame, (sx, sy), 20, dot_color, -1)
            cv2.circle(cal_frame, (sx, sy), 20, (255, 255, 255), 2, cv2.LINE_AA)

            if samples:
                cv2.putText(cal_frame, f"ornek: {len(samples)}", (sx - 40, sy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1, cv2.LINE_AA)

            cv2.imshow(cal_win, cal_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(cal_win)
                return None

            if elapsed >= CAL_WAIT + CAL_COLLECT:
                break

        if len(samples) >= 10:
            used = _best_samples(samples)
            features_list.append(_trim_mean(used))
            screen_pts.append((sx, sy))
            pt_idx += 1
        else:
            pt_idx += 1

    cv2.destroyWindow(cal_win)

    if len(features_list) < 5:
        return None

    return features_list, screen_pts


# ─────────────────────────────────────────────
# Ana döngü
# ─────────────────────────────────────────────

def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gaze_mapper    = GazeMapper()
    blink_detector = BlinkDetector()
    frame_times    = collections.deque(maxlen=30)
    mouse_active   = True
    click_enabled  = True

    cal = run_calibration(cap, face_landmarker)
    if cal is None:
        cap.release()
        face_landmarker.close()
        return
    gaze_mapper.calibrate(*cal)

    while True:
        frame, result, orig_w, orig_h = process_frame(cap, face_landmarker)
        if frame is None:
            break

        frame_times.append(time.perf_counter())
        fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
               if len(frame_times) > 1 else 0.0)

        blinking = False

        if result.face_landmarks and result.facial_transformation_matrixes:
            lms = result.face_landmarks[0]

            left_pts       = [lm_to_px(lms[i], orig_w, orig_h) for i in LEFT_EYE]
            right_pts      = [lm_to_px(lms[i], orig_w, orig_h) for i in RIGHT_EYE]
            left_iris_pts  = [lm_to_px(lms[i], orig_w, orig_h) for i in LEFT_IRIS]
            right_iris_pts = [lm_to_px(lms[i], orig_w, orig_h) for i in RIGHT_IRIS]

            lx0, ly0, lx1, ly1 = eye_bbox(left_pts)
            rx0, ry0, rx1, ry1 = eye_bbox(right_pts)
            cv2.rectangle(frame, (lx0, ly0), (lx1, ly1), (0, 220, 0), 2)
            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 220, 0), 2)
            draw_iris(frame, left_iris_pts)
            draw_iris(frame, right_iris_pts)

            ear = compute_ear(lms)
            now = frame_times[-1]
            blinking, double_blink = blink_detector.update(ear, now)

            if double_blink and click_enabled:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                user32.mouse_event(MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)

            ear_color = (0, 0, 200) if blinking else (0, 200, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, orig_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1, cv2.LINE_AA)

            feats = compute_gaze_features(lms, result.facial_transformation_matrixes[0])
            pos   = gaze_mapper.map(feats, blinking=blinking)
            if pos and mouse_active:
                user32.SetCursorPos(pos[0], pos[1])

            # TPS durum göstergesi
            tps_label = "TPS" if gaze_mapper.tps is not None else "POLY"
            cv2.putText(frame, f"MODEL: {tps_label}", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 60), 1, cv2.LINE_AA)

            if double_blink and click_enabled:
                status_text, status_color = "TIKLADI!", (0, 220, 255)
            elif blinking:
                status_text, status_color = "KIRPMA", (0, 80, 220)
            elif not mouse_active:
                status_text, status_color = "DURDURULDU", (0, 165, 255)
            else:
                status_text, status_color = "ACIK", (0, 220, 0)
        else:
            gaze_mapper.map(None)
            status_text  = "YUZ YOK"
            status_color = (0, 0, 220)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GOZ: {status_text}", (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)
        click_label = "ACIK" if click_enabled else "KAPALI"
        click_color = (0, 200, 200) if click_enabled else (80, 80, 80)
        cv2.putText(frame, f"2x KIRPMA TIKLAMA: {click_label}", (10, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, click_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "q:cik  c:kalibrasyon  p:durdur/devam  b:tiklama ac/kapat",
                    (10, orig_h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imshow("Goz Algilama", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            new_cal = run_calibration(cap, face_landmarker)
            if new_cal:
                gaze_mapper.calibrate(*new_cal)
        elif key == ord('p'):
            mouse_active = not mouse_active
        elif key == ord('b'):
            click_enabled = not click_enabled

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()


if __name__ == "__main__":
    main()