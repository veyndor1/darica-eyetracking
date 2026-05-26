import collections
import ctypes
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# --- Landmark indeksleri ---
LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Göz köşe / üst-alt noktaları
L_OUTER, L_INNER, L_TOP, L_BOT = 33, 133, 159, 145
R_OUTER, R_INNER, R_TOP, R_BOT = 263, 362, 386, 374

# EAR (göz açıklığı) için yatay + dikey noktalar
# Sol: yatay=(33,133), dikey=(159,145, 158,153)
# Sağ: yatay=(263,362), dikey=(386,374, 385,380)

# ════════════════════════════════════════════════════════════════
# AYARLANAB�L�R PARAMETRELER — sadece bu bloğu düzenle
# ════════════════════════════════════════════════════════════════

# ── Kamera çözünürlüğü ───────────────────────────────────────────
# Yüz tespiti için kareyi bu genişliğe küçültür; küçültünce işlem hızlanır.
# Azalt (örn. 480) → daha hızlı ama daha az hassas
# Artır (örn. 960) → daha yavaş ama daha hassas
PROCESS_WIDTH = 640

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

user32   = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

# ── Kalibrasyon noktaları (ekran oranı 0.0–1.0) ──────────────────
# Her tuple (yatay, dikey). 0.0=sol/üst kenar, 1.0=sağ/alt kenar.
# Noktaları kenara yaklaştırırsan (örn. 0.04) imlecin köşelere
# ulaşması kolaylaşır; merkezi kalabalıklaştırırsan hassasiyet artar.
CAL_POINTS_REL = [
    (0.50, 0.50),
    (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
    (0.08, 0.50),               (0.92, 0.50),
    (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
    (0.25, 0.25), (0.75, 0.25),
    (0.25, 0.75), (0.75, 0.75),
]

# ── Kalibrasyon süresi ────────────────────────────────────────────
# CAL_WAIT   : noktaya baktıktan sonra bekleme süresi (saniye).
#              Artır → göz daha iyi yerleşir, kalibrasyon yavaşlar.
# CAL_COLLECT: örnek toplama süresi (saniye).
#              Artır → daha güvenilir kalibrasyon, daha uzun sürer.
CAL_WAIT    = 1.8
CAL_COLLECT = 3.0

# ── Kalibrasyon regülasyonu ───────────────────────────────────────
# Ridge regresyon katsayısı. Büyük değer → model daha sert/kaba,
# küçük değer → model daha esnek ama az noktada kararsız olabilir.
# Önerilen aralık: 0.001 – 0.1
CAL_RIDGE   = 0.02

# ── Göz kırpma eşiği (EAR) ───────────────────────────────────────
# Göz açıklık oranı bu değerin ALTINA düşünce "kırpma" sayılır.
# Artır (örn. 0.20) → daha kolay tetiklenir, yanlış kırpma artar.
# Azalt (örn. 0.14) → daha zor tetiklenir, gerçek kırpmalar kaçabilir.
EAR_BLINK   = 0.17

# ── İmleç konum düzeltmesi ───────────────────────────────────────
# İmleç sürekli sola kayıyorsa → GAZE_X_OFFSET değerini artır (örn. +60).
# İmleç sürekli sağa kayıyorsa → GAZE_X_OFFSET değerini azalt (örn. -60).
# İmleç sürekli yukarı kayıyorsa → GAZE_Y_OFFSET değerini azalt (örn. -40).
# İmleç sürekli aşağı kayıyorsa → GAZE_Y_OFFSET değerini artır (örn. +40).
GAZE_X_OFFSET        = 0
GAZE_Y_OFFSET        = +20

# ── Y ekseni yönü ────────────────────────────────────────────────
# Yukarı baktığında imleç aşağı gidiyorsa bu değeri 1 yap.
# Yukarı baktığında imleç yukarı gidiyorsa (doğru) -1 bırak.
GAZE_Y_FLIP          = -1

# ── Çift kırpma zaman penceresi ──────────────────────────────────
# İki kırpma arasındaki maksimum süre (saniye). Bu sürede iki kırpma
# gerçekleşirse sol tıklama tetiklenir.
# Artır → daha geniş zaman aralığı, yanlış tıklama riski artar.
# Azalt → daha hızlı çift kırpma gerekir.
DOUBLE_BLINK_MAX_GAP = 1.5

# ── Ani kafa hareketi kilidi ──────────────────────────────────────
# Kafanın frame başına bu kadar (radyan) döndüğünde imleç dondurulur.
# Azalt (örn. 0.07) → küçük kafa hareketlerinde bile dondurur.
# Artır (örn. 0.20) → sadece çok ani hareketlerde dondurur.
HEAD_DELTA_GATE      = 0.12

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

# ── Blink rollback buffer ─────────────────────────────────────────
# EAR eşiği geç tetiklenir; o sırada iris landmark'ları zaten bozulmuştur
# (Google Project Gameface blog, 2023). Blink başlangıcında imleci
# buffer'da BLINK_ROLLBACK_FRAMES kadar frame öncesine geri sar.
#   Artır (örn. 5) → daha güvenli geri sarma, ufak gecikme.
#   Azalt (örn. 2) → daha hızlı tepki, bazı kırpmalarda kalıntı sıçrama.
BLINK_ROLLBACK_FRAMES = 3

# Erken blink tespiti: son 4 frame içindeki EAR düşüşü bu eşiği
# (negatif değer) geçerse imleç dondurulur ve geri sarılır.
# Azalt (örn. -0.08) → sadece sert kırpmalarda tetiklenir.
# Artır (örn. -0.04) → küçük kapanmalarda bile dondurur.
BLINK_EAR_DERIV_DROP  = -0.06

# EAR bu seviyeye çıkıp 2 frame stable olunca kırpma bitti sayılır
# (Soukupová & Čech 2016: 0.20; PyImageSearch: 0.30 referansları).
EAR_REOPEN            = 0.22

# ── Cursor düzlemi One-Euro ───────────────────────────────────────
# Casiez et al. 2012 (CHI '12, doi:10.1145/2207676.2208639):
# tek katmanlı One-Euro Kalman'dan daha düşük SEM verir. İki filtreyi
# zincirlemek (cascade) double-smoothing lag yaratır ve blink-jump
# bug'ının ikinci ana sebebidir → Kalman kaldırıldı, tek katman kaldı.
CURSOR_OEF_MINCUTOFF  = 1.0
CURSOR_OEF_BETA       = 0.05

# ── Ölü bölge (cursor-velocity tabanlı) ───────────────────────────
# Sabit ölü bölge (piksel). Bu kadardan küçük hareketler yok sayılır.
# Artır → imleç daha az titrer ama hassasiyet düşer.
DEAD_ZONE            = 7
ADAPTIVE_DEAD_BASE   = DEAD_ZONE
ADAPTIVE_DEAD_BONUS  = 10
ADAPTIVE_DEAD_SPEED  = 12.0

# ════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# One Euro Filter — iris özelliklerini kaynağında filtreler
# ─────────────────────────────────────────────

class OneEuroFilter:
    """
    Hız-adaptif düşük geçiren filtre (Casiez et al. 2012).

    - Sabit bakış (yavaş/sıfır hız)  → güçlü yumuşatma, titreme yok
    - Kasıtlı hareket (yüksek hız)   → zayıf yumuşatma, gecikme yok

    Parametreler:
      min_cutoff : Hz — düşükse daha güçlü statik yumuşatma
      beta       : hız katsayısı — yükselirse hızlı hareket daha az filtrelenir
      d_cutoff   : türev filtresinin kesim frekansı (Hz)
    """
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
        dt = max(t - self._t, 1e-6)
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
    """Eye Aspect Ratio — her iki göz ortalaması. Kırpma tespiti için."""
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
    """
    MediaPipe facial_transformation_matrixes'ten yaw ve pitch açılarını çıkar (radyan).
    Dönüş matrisi (3x3) → cv2.Rodrigues ile rotasyon vektörü → pitch/yaw.
    """
    mat = np.array(transform_matrix_data.data, dtype=float).reshape(4, 4)
    R   = mat[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    pitch = float(rvec[0, 0])   # yukarı/aşağı kafa hareketi
    yaw   = float(rvec[1, 0])   # sola/sağa kafa hareketi
    return yaw, pitch


def compute_gaze_features(lms, transform_matrix_data):
    """
    6 özellikli gaze vektörü döner:
        [yaw, pitch, left_rx, right_rx, left_ry, right_ry]

    Sol ve sağ göz iris konumları AYRI döndürülür — GazeMapper içinde
    medyan + MAD outlier reddi yapılır (Pupil Labs / MMransac yaklaşımı).
    Tek-göz ortalaması head tilt'e karşı asimetrik bias üretiyordu.
    """
    yaw, pitch = get_head_angles(transform_matrix_data)

    def _rel_x(iris_idx, outer_idx, inner_idx):
        ew = abs(lms[inner_idx].x - lms[outer_idx].x)
        if ew < 1e-5:
            return None
        cx = (lms[outer_idx].x + lms[inner_idx].x) / 2.0
        return (lms[iris_idx].x - cx) / ew

    def _rel_y(iris_idx, top_idx, bot_idx):
        eh = abs(lms[top_idx].y - lms[bot_idx].y)
        if eh < 1e-5:
            return None
        cy = (lms[top_idx].y + lms[bot_idx].y) / 2.0
        return (lms[iris_idx].y - cy) / eh

    left_rx  = _rel_x(468, L_OUTER, L_INNER)
    right_rx = _rel_x(473, R_OUTER, R_INNER)
    left_ry  = _rel_y(468, L_TOP, L_BOT)
    right_ry = _rel_y(473, R_TOP, R_BOT)

    # Eksik gözü diğer gözle doldur — ikisi de yoksa frame'i at
    if left_rx is None and right_rx is None:
        return None
    if left_ry is None and right_ry is None:
        return None
    if left_rx is None:  left_rx  = right_rx
    if right_rx is None: right_rx = left_rx
    if left_ry is None:  left_ry  = right_ry
    if right_ry is None: right_ry = left_ry

    return [
        yaw, pitch,
        left_rx, right_rx,
        left_ry * GAZE_Y_FLIP, right_ry * GAZE_Y_FLIP,
    ]


def process_frame(cap, face_landmarker):
    """Kameradan frame yakala ve yüz tespiti yap."""
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
# GazeMapper — Ridge + PolynomialFeatures + head-pose decoupling + blink rollback
# ─────────────────────────────────────────────
#
# Drift'in kök sebebi: sıradan lineer regresyonda yaw ↔ iris_rel_x
# multicollinearity'i var; (X^T X)^-1 patlar, küçük gürültü büyük katsayıya
# dönüşür → sistematik tek-yön drift. Ridge: β̂ = (X^T X + αI)^-1 X^T y.
# Polynomial degree=2: gözün küresel projeksiyon geometrisini modeller.
# Head-pose decoupling: kafa açısı iris vektöründen çıkarılır → korelasyon
# yapısı kırılır.
#
# Blink-jump'ın kök sebebi: MediaPipe iris landmark'ları kapanan göz kapağıyla
# birlikte aşağı çekilir (Google Project Gameface blog, 2023). EAR eşiği geç
# tetiklenir; landmarks zaten 2-3 frame bozulmuştur. Çözüm:
#   1. EAR türevi + mutlak eşik kombinasyonu → erken tespit
#   2. Rolling buffer'dan BLINK_ROLLBACK_FRAMES öncesine geri sar
#   3. Filtrelere ölçüm sokma → state bozulmasın

class GazeMapper:
    """
    Pipeline (her frame):
      iris_rel = (sol + sağ) / 2   (medyan-MAD outlier reddi)
      iris_comp_x = iris_rel_x - k_yaw * yaw
      iris_comp_y = iris_rel_y - k_pitch * pitch
      F = poly([iris_comp_x, iris_comp_y, yaw, pitch], degree=2)
      raw_x = Ridge_X.predict(F)
      raw_y = Ridge_Y.predict(F)
      sx, sy = normalize(raw_x, raw_y)
      cursor = OneEuro(sx, sy)
    """
    def __init__(self):
        self.poly    = PolynomialFeatures(degree=2, include_bias=True)
        self.model_x = None   # Ridge — None ise kalibre edilmemiş
        self.model_y = None
        self.k_yaw   = 0.0    # head-pose decoupling katsayıları
        self.k_pitch = 0.0
        # Çıkış normalizasyonu: kalibrasyon aralığını tam ekrana ger
        self._nx_a = 1.0; self._nx_b = 0.0
        self._ny_a = 1.0; self._ny_b = 0.0
        # Tek katmanlı One-Euro filtre (cursor düzleminde)
        self._oef_x = OneEuroFilter(min_cutoff=CURSOR_OEF_MINCUTOFF, beta=CURSOR_OEF_BETA)
        self._oef_y = OneEuroFilter(min_cutoff=CURSOR_OEF_MINCUTOFF, beta=CURSOR_OEF_BETA)
        # Blink rollback buffer'ı (en eski = rollback hedefi)
        self.cursor_buffer = deque(maxlen=BLINK_ROLLBACK_FRAMES + 1)
        self.ear_history   = deque(maxlen=5)
        self.in_blink      = False
        self.frozen_pos    = None
        self._reopen_count = 0
        # Diğer state
        self.sent_pos    = (SCREEN_W // 2, SCREEN_H // 2)
        self._prev_yaw   = None
        self._prev_pitch = None
        self._last_pos   = None   # önceki cursor — adaptif ölü bölge için

    # --- yardımcılar ----------------------------------------------------

    @staticmethod
    def _mean_iris(f):
        """Sol/sağ iris değerlerinin ortalaması. Outlier reddi calibrate'te."""
        _, _, l_rx, r_rx, l_ry, r_ry = f
        return (l_rx + r_rx) / 2.0, (l_ry + r_ry) / 2.0

    @staticmethod
    def _mad_filter(arr_2d, k=1.5):
        """Medyan ± k×MAD dışındaki satırların boolean mask'i (True = tut)."""
        med = np.median(arr_2d, axis=0)
        mad = np.median(np.abs(arr_2d - med), axis=0) + 1e-9
        return np.all(np.abs(arr_2d - med) <= k * mad * 4.0, axis=1)

    def _build_feature_vec(self, yaw, pitch, iris_rel_x, iris_rel_y):
        iris_comp_x = iris_rel_x - self.k_yaw   * yaw
        iris_comp_y = iris_rel_y - self.k_pitch * pitch
        return np.array([iris_comp_x, iris_comp_y, yaw, pitch], dtype=float)

    # --- kalibrasyon ----------------------------------------------------

    def calibrate(self, features_list, screen_pts):
        # features_list elemanı: [yaw, pitch, l_rx, r_rx, l_ry, r_ry]
        raw = np.array(features_list, dtype=float)
        yaws    = raw[:, 0]
        pitches = raw[:, 1]
        iris_xs = (raw[:, 2] + raw[:, 3]) / 2.0
        iris_ys = (raw[:, 4] + raw[:, 5]) / 2.0

        # Head-pose decoupling katsayıları: cov(iris, head) / var(head)
        # Kafanın hareket aralığı çok daraysa katsayı 0'a yakın çıkar → no-op
        var_yaw   = float(np.var(yaws))
        var_pitch = float(np.var(pitches))
        self.k_yaw   = float(np.cov(iris_xs, yaws)[0, 1] / var_yaw)   if var_yaw   > 1e-6 else 0.0
        self.k_pitch = float(np.cov(iris_ys, pitches)[0, 1] / var_pitch) if var_pitch > 1e-6 else 0.0

        # Feature vektörlerini decoupled olarak kur
        F = np.array([
            self._build_feature_vec(yaws[i], pitches[i], iris_xs[i], iris_ys[i])
            for i in range(len(features_list))
        ], dtype=float)
        bx = np.array([p[0] for p in screen_pts], dtype=float)
        by = np.array([p[1] for p in screen_pts], dtype=float)

        Fpoly = self.poly.fit_transform(F)
        self.model_x = Ridge(alpha=CAL_RIDGE)
        self.model_y = Ridge(alpha=CAL_RIDGE)
        self.model_x.fit(Fpoly, bx)
        self.model_y.fit(Fpoly, by)

        # Çıkış normalizasyonu: model tahminlerinin min–max aralığını tam
        # ekrana ger. Kalibrasyon noktaları %8–%92 arasında olduğundan
        # kullanıcı kenar noktasına bakınca cursor gerçek kenara ulaşır.
        pred_x = self.model_x.predict(Fpoly)
        pred_y = self.model_y.predict(Fpoly)
        px_lo, px_hi = float(pred_x.min()), float(pred_x.max())
        py_lo, py_hi = float(pred_y.min()), float(pred_y.max())
        if px_hi > px_lo:
            self._nx_a = SCREEN_W / (px_hi - px_lo)
            self._nx_b = -self._nx_a * px_lo
        else:
            self._nx_a, self._nx_b = 1.0, 0.0
        if py_hi > py_lo:
            self._ny_a = SCREEN_H / (py_hi - py_lo)
            self._ny_b = -self._ny_a * py_lo
        else:
            self._ny_a, self._ny_b = 1.0, 0.0

        # Tüm runtime state'i sıfırla
        self.sent_pos    = (SCREEN_W // 2, SCREEN_H // 2)
        self._prev_yaw   = None
        self._prev_pitch = None
        self._last_pos   = None
        self._oef_x.reset()
        self._oef_y.reset()
        self.cursor_buffer.clear()
        self.ear_history.clear()
        self.in_blink      = False
        self.frozen_pos    = None
        self._reopen_count = 0

    # --- inference -------------------------------------------------------

    def _early_blink(self, ear):
        """EAR mutlak eşik VEYA 4-frame türev düşüşü → True."""
        self.ear_history.append(ear)
        if ear < EAR_BLINK:
            return True
        if len(self.ear_history) >= 4:
            deriv = self.ear_history[-1] - self.ear_history[-4]
            if deriv < BLINK_EAR_DERIV_DROP:
                return True
        return False

    def map(self, features, ear=None, blinking=False):
        if self.model_x is None or features is None:
            pos = self.sent_pos
            return pos

        # ── Blink yönetimi (EAR tabanlı erken tespit + rollback) ─────────
        if ear is not None:
            unstable = blinking or self._early_blink(ear)
        else:
            unstable = blinking

        if unstable:
            if not self.in_blink:
                # Blink BAŞLADI — buffer'ın en eskisine (BLINK_ROLLBACK_FRAMES öncesi) geri sar
                if self.cursor_buffer:
                    self.frozen_pos = self.cursor_buffer[0]
                else:
                    self.frozen_pos = self.sent_pos
                self.in_blink = True
                self._reopen_count = 0
            # Blink esnasında filtreye ölçüm SOKMA, buffer'a EKLEME
            self.sent_pos = self.frozen_pos
            return self.frozen_pos

        # Blink bitiş hysteresis: EAR > EAR_REOPEN ve 2 frame stable
        if self.in_blink:
            if ear is not None and ear > EAR_REOPEN:
                self._reopen_count += 1
                if self._reopen_count >= 2:
                    self.in_blink = False
                    # Filtreyi son güvenilir cursor'la reset et
                    self._oef_x.reset()
                    self._oef_y.reset()
                else:
                    return self.frozen_pos
            else:
                self._reopen_count = 0
                return self.frozen_pos

        yaw, pitch = features[0], features[1]
        iris_rel_x, iris_rel_y = self._mean_iris(features)

        # Ani kafa hareketi → imleci dondur (filtreyi de besleme)
        if self._prev_yaw is not None:
            if (abs(yaw - self._prev_yaw) > HEAD_DELTA_GATE or
                    abs(pitch - self._prev_pitch) > HEAD_DELTA_GATE):
                self._prev_yaw, self._prev_pitch = yaw, pitch
                return self.sent_pos
        self._prev_yaw, self._prev_pitch = yaw, pitch

        # Decoupled feature → polynomial expand → Ridge predict
        F = self._build_feature_vec(yaw, pitch, iris_rel_x, iris_rel_y).reshape(1, -1)
        Fpoly = self.poly.transform(F)
        raw_x = float(self.model_x.predict(Fpoly)[0])
        raw_y = float(self.model_y.predict(Fpoly)[0])

        # Normalize + offset + clip
        sx = float(np.clip(self._nx_a * raw_x + self._nx_b + GAZE_X_OFFSET, 0, SCREEN_W - 1))
        sy = float(np.clip(self._ny_a * raw_y + self._ny_b + GAZE_Y_OFFSET, 0, SCREEN_H - 1))

        # Tek katmanlı One-Euro (cursor düzleminde)
        t = time.perf_counter()
        fx = self._oef_x.update(sx, t)
        fy = self._oef_y.update(sy, t)
        smoothed = (int(np.clip(fx, 0, SCREEN_W - 1)),
                    int(np.clip(fy, 0, SCREEN_H - 1)))

        # Adaptif ölü bölge — cursor delta'sından hız tahmini
        if self._last_pos is not None:
            speed = math.hypot(smoothed[0] - self._last_pos[0],
                               smoothed[1] - self._last_pos[1])
        else:
            speed = 0.0
        scale = max(0.0, 1.0 - speed / ADAPTIVE_DEAD_SPEED)
        dead  = ADAPTIVE_DEAD_BASE + int(ADAPTIVE_DEAD_BONUS * scale)

        dx = abs(smoothed[0] - self.sent_pos[0])
        dy = abs(smoothed[1] - self.sent_pos[1])
        if dx < dead and dy < dead:
            # Hareket ölü bölgenin içinde — pozisyonu sabit tut
            self._last_pos = smoothed
            self.cursor_buffer.append(self.sent_pos)
            return self.sent_pos

        self.sent_pos  = smoothed
        self._last_pos = smoothed
        self.cursor_buffer.append(smoothed)
        return smoothed


# ─────────────────────────────────────────────
# Çift kırpma tespiti
# ─────────────────────────────────────────────

class BlinkDetector:
    """
    EAR akışından tek ve çift göz kırpmayı ayırt eder.

    Mantık:
      - EAR < EAR_BLINK için en az 2 ardışık frame → kırpma BAŞLADI
      - EAR tekrar eşiğin üzerine çıkınca → kırpma BİTTİ (olay)
      - İki olay DOUBLE_BLINK_MAX_GAP saniye içindeyse → çift kırpma
    """
    def __init__(self):
        self.blink_frames  = 0
        self.in_blink      = False        # o an kırpıyor mu
        self.blink_times   = collections.deque(maxlen=3)  # son kırpma bitiş zamanları

    def update(self, ear, now):
        """
        Döner: (is_blinking: bool, double_blink: bool)
        """
        double_blink = False

        if ear < EAR_BLINK:
            self.blink_frames += 1
            is_blinking = self.blink_frames >= 2
        else:
            # Kırpma bitti mi?
            if self.in_blink and self.blink_frames >= 2:
                self.blink_times.append(now)
                # Son iki kırpma olayı yeterince yakın mı?
                if (len(self.blink_times) >= 2
                        and now - self.blink_times[-2] <= DOUBLE_BLINK_MAX_GAP):
                    double_blink = True
                    self.blink_times.clear()   # tekrar tetiklenmesin
            self.blink_frames = 0
            is_blinking = False

        self.in_blink = is_blinking
        return is_blinking, double_blink


# ─────────────────────────────────────────────
# Kalibrasyon ekranı
# ─────────────────────────────────────────────

def _trim_mean(samples, trim=0.15):
    """Aşırı değerleri at (alt ve üst %trim), kalanı ortala."""
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
    """
    Kalibrasyon kalitesini artırmak için iki adım:
    1. Sadece son %55'i kullan — göz noktaya tam yerleşmiş olur.
    2. Iris hareketi düşük olan kareleri tut (saccade artığı kareler atılır).

    Feature layout: [yaw, pitch, l_rx, r_rx, l_ry, r_ry]
    Iris hızı için sol+sağ ortalaması kullanılır.
    """
    # Adım 1: son %55
    cutoff = int(len(samples) * 0.45)
    late   = samples[cutoff:] if len(samples) > 12 else samples

    # Adım 2: durağanlık filtresi (iris ortalaması üzerinden)
    if len(late) < 4:
        return late
    arr = np.array(late, dtype=float)
    iris_x = (arr[:, 2] + arr[:, 3]) / 2.0
    iris_y = (arr[:, 4] + arr[:, 5]) / 2.0
    vel = np.sqrt(np.diff(iris_x) ** 2 + np.diff(iris_y) ** 2)
    thresh = np.percentile(vel, 60)   # en durağan %60'ını tut
    keep   = [0] + [i + 1 for i, v in enumerate(vel) if v <= thresh]
    stable = [late[i] for i in keep]
    return stable if len(stable) >= 5 else late


def run_calibration(cap, face_landmarker):
    """
    Tam ekran 9 noktalı kalibrasyon.
    Döner: (features_list, screen_pts)  ya da  None (iptal).
    """
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
                if ear > EAR_BLINK:   # göz açık
                    feats = compute_gaze_features(lms, result.facial_transformation_matrixes[0])
                    if feats:
                        samples.append(feats)

            # --- Çizim ---
            cal_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

            msg = f"Noktaya bakin  ({pt_idx + 1} / {len(CAL_POINTS_REL)})"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(cal_frame, msg, ((SCREEN_W - tw) // 2, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
            hint = "Noktaya bakarken kafanizi cok hafif oynatabilirsiniz"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.putText(cal_frame, hint, ((SCREEN_W - hw) // 2, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 160, 200), 1, cv2.LINE_AA)
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

            # Örnek sayısı göster
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
            # Durağan + geç örnekleri seç, sonra trim_mean uygula
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
    click_enabled  = True   # çift kırpma = tıklama özelliği

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

            # Göz kutuları ve iris çiz
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

            # Kırpma tespiti
            ear = compute_ear(lms)
            now = frame_times[-1]
            blinking, double_blink = blink_detector.update(ear, now)

            # Çift kırpma → sol tıklama (özellik açıksa)
            if double_blink and click_enabled:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                user32.mouse_event(MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)

            # EAR göstergesi
            ear_color = (0, 0, 200) if blinking else (0, 200, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, orig_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1, cv2.LINE_AA)

            # Mouse hareketi (EAR'ı mapper'a verip erken blink tespitini etkinleştir)
            feats = compute_gaze_features(lms, result.facial_transformation_matrixes[0])
            pos   = gaze_mapper.map(feats, ear=ear, blinking=blinking)
            if pos and mouse_active:
                user32.SetCursorPos(pos[0], pos[1])

            if double_blink and click_enabled:
                status_text, status_color = "TIKLADI!", (0, 220, 255)
            elif blinking:
                status_text, status_color = "KIRPMA", (0, 80, 220)
            elif not mouse_active:
                status_text, status_color = "DURDURULDU", (0, 165, 255)
            else:
                status_text, status_color = "ACIK", (0, 220, 0)
        else:
            gaze_mapper.map(None)   # cursor son sent_pos'ta kalsın
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
