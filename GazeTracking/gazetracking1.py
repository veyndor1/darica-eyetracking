import collections
import ctypes
import ctypes.wintypes
import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np

# ─── Landmark indeksleri ──────────────────────────────────────────────────────
LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_IRIS  = [468,469,470,471,472]
RIGHT_IRIS = [473,474,475,476,477]

PROCESS_WIDTH = 640
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Model dosyası önce script klasöründe, yoksa üst klasörde (proje kökü) aranır.
MODEL_PATH = os.path.join(_SCRIPT_DIR, "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "face_landmarker.task")

user32   = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

# ─── Kalibrasyon sabitleri ────────────────────────────────────────────────────
CAL_POINTS_REL = [
    (0.50,0.50),
    (0.08,0.08),(0.92,0.08),
    (0.08,0.92),(0.92,0.92),
    (0.50,0.08),(0.50,0.92),
    (0.08,0.50),(0.92,0.50),
    (0.29,0.29),(0.71,0.29),
    (0.29,0.71),(0.71,0.71),
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
HEAD_DELTA_GATE      = 0.22
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004
VEL_DAMP_THRESHOLD   = 0.6
ADAPTIVE_DEAD_BASE   = DEAD_ZONE
ADAPTIVE_DEAD_BONUS  = 5
ADAPTIVE_DEAD_SPEED  = 12.0
TPS_IRIS_WEIGHT      = 1.5

# ─── Unicode tuş gönderici ───────────────────────────────────────────────────
INPUT_KEYBOARD    = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP   = 0x0002
VK_BACK           = 0x08
VK_RETURN         = 0x0D
VK_SPACE          = 0x20

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         ctypes.c_ushort),
        ("wScan",       ctypes.c_ushort),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_uint64),
    ]

class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki",   _KEYBDINPUT),
        ("_pad", ctypes.c_ubyte * 8),
    ]

def send_unicode_char(ch: str):
    """Aktif uygulamaya bir Unicode karakter gönder."""
    code = ord(ch)
    for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
        inp = _INPUT()
        inp.type       = INPUT_KEYBOARD
        inp.ki.wScan   = code
        inp.ki.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

def send_vk(vk: int):
    """Virtual key gönder (Backspace, Enter vb.)."""
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


# ─── One Euro Filter ─────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, min_cutoff=0.9, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x = None; self._dx = 0.0; self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def update(self, x, t=None):
        if t is None: t = time.perf_counter()
        if self._x is None:
            self._x, self._t = float(x), t; return self._x
        dt       = max(t - self._t, 1e-6)
        dx       = (x - self._x) / dt
        a_d      = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        a        = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x  = a * float(x) + (1.0 - a) * self._x
        self._t  = t
        return self._x

    def reset(self):
        self._x = None; self._dx = 0.0; self._t = None


# ─── Yardımcı fonksiyonlar ───────────────────────────────────────────────────
def lm_to_px(lm, w, h): return (int(lm.x*w), int(lm.y*h))

def eye_bbox(points, pad_x=0.25, pad_y=0.40):
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys)
    pw=int((x1-x0)*pad_x); ph=int((y1-y0)*pad_y)
    return x0-pw,y0-ph,x1+pw,y1+ph

def draw_iris(frame, iris_px):
    cx,cy=iris_px[0]
    radius=int(np.mean([np.hypot(p[0]-cx,p[1]-cy) for p in iris_px[1:]])) if len(iris_px)>1 else 8
    radius=max(radius,2)
    cv2.circle(frame,(cx,cy),radius,(0,0,220),2)
    cv2.circle(frame,(cx,cy),2,(255,230,0),-1)

def compute_ear(lms):
    def _ear(h1,h2,v1,v2,v3,v4):
        horiz=math.hypot(lms[h1].x-lms[h2].x,lms[h1].y-lms[h2].y)
        if horiz<1e-6: return 1.0
        return (math.hypot(lms[v1].x-lms[v2].x,lms[v1].y-lms[v2].y)+
                math.hypot(lms[v3].x-lms[v4].x,lms[v3].y-lms[v4].y))/(2.0*horiz)
    return (_ear(33,133,159,145,158,153)+_ear(263,362,386,374,385,380))/2.0

def get_head_angles(tm):
    mat=np.array(tm.data,dtype=float).reshape(4,4)
    rvec,_=cv2.Rodrigues(mat[:3,:3])
    return float(rvec[1,0]), float(rvec[0,0])   # yaw, pitch

def compute_gaze_features(lms, tm):
    yaw, pitch = get_head_angles(tm)
    def _im(idx):
        n=len(idx); return sum(lms[i].x for i in idx)/n, sum(lms[i].y for i in idx)/n
    def _es(idx):
        xs=[lms[i].x for i in idx]; ys=[lms[i].y for i in idx]
        return min(xs),max(xs),min(ys),max(ys)
    lx0,lx1,ly0,ly1=_es(LEFT_EYE);  lix,liy=_im(LEFT_IRIS);  lew=lx1-lx0; leh=ly1-ly0
    rx0,rx1,ry0,ry1=_es(RIGHT_EYE); rix,riy=_im(RIGHT_IRIS); rew=rx1-rx0; reh=ry1-ry0
    lrx=(lix-(lx0+lx1)/2)/lew if lew>1e-5 else None
    rrx=(rix-(rx0+rx1)/2)/rew if rew>1e-5 else None
    if lrx is not None and rrx is not None: irx=(lrx+rrx)/2
    elif lrx is not None: irx=lrx
    elif rrx is not None: irx=rrx
    else: return None
    lry=(liy-(ly0+ly1)/2)/leh if leh>1e-5 else None
    rry=(riy-(ry0+ry1)/2)/reh if reh>1e-5 else None
    if lry is not None and rry is not None: iry=(lry+rry)/2
    elif lry is not None: iry=lry
    elif rry is not None: iry=rry
    else: return None
    verg=(lrx-rrx) if (lrx is not None and rrx is not None) else 0.0
    return [yaw, pitch, irx, iry*GAZE_Y_FLIP, verg]

def process_frame(cap, fl):
    ret, frame = cap.read()
    if not ret: return None,None,0,0
    h,w=frame.shape[:2]
    sc=PROCESS_WIDTH/w if w>PROCESS_WIDTH else 1.0
    small=cv2.resize(frame,(int(w*sc),int(h*sc))) if sc<1.0 else frame
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
    return frame, fl.detect(mp_img), w, h


# ─── Kalman Filtresi ─────────────────────────────────────────────────────────
class KalmanGaze:
    def __init__(self, process_noise=20.0, measure_noise=600.0):
        dt=1/30
        self.x=np.array([[SCREEN_W/2.],[SCREEN_H/2.],[0.],[0.]])
        self.P=np.eye(4)*2000.
        self.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
        self.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        self.Q=np.eye(4)*process_noise
        self.R=np.eye(2)*measure_noise

    def update(self, meas):
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
        z=np.array([[float(meas[0])],[float(meas[1])]])
        y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y; self.P=(np.eye(4)-K@self.H)@self.P
        self._damp(); return self._clamp()

    def predict_only(self):
        self.x[2,0]=0.; self.x[3,0]=0.           # FIX 1: hız sıfırla
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
        self._damp(); return self._clamp()

    def _damp(self):
        if abs(self.x[2,0])<VEL_DAMP_THRESHOLD: self.x[2,0]=0.
        if abs(self.x[3,0])<VEL_DAMP_THRESHOLD: self.x[3,0]=0.

    def _clamp(self):
        return (int(np.clip(self.x[0,0],0,SCREEN_W-1)),
                int(np.clip(self.x[1,0],0,SCREEN_H-1)))


# ─── Polinom & TPS ───────────────────────────────────────────────────────────
def _feat_x(y, ix): return np.array([y,ix,y*ix,y**2,ix**2,1.],float)
def _feat_y(p, iy): return np.array([p,iy,p*iy,p**2,iy**2,1.],float)
def _ridge(A,b,lam):
    M=A.T@A; M+=lam*np.eye(M.shape[0]); return np.linalg.solve(M,A.T@b)

class ThinPlateSpline:
    @staticmethod
    def _rbf(r): return r*r*math.log(r+1e-10)
    def fit(self, src, dx, dy, lam=1e-4):
        self._s=np.array(src,float); N=len(self._s)
        K=np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                r=math.hypot(self._s[i,0]-self._s[j,0],self._s[i,1]-self._s[j,1])
                K[i,j]=K[j,i]=self._rbf(r)
        P=np.hstack([np.ones((N,1)),self._s])
        M=np.block([[K+lam*np.eye(N),P],[P.T,np.zeros((3,3))]])
        rx=np.concatenate([np.array(dx,float),[0,0,0]])
        ry=np.concatenate([np.array(dy,float),[0,0,0]])
        sol=np.linalg.solve(M,np.column_stack([rx,ry]))
        self._w=sol[:N]; self._a=sol[N:]
    def predict(self, gx, gy):
        v=np.array([self._rbf(math.hypot(gx-s[0],gy-s[1])) for s in self._s])
        o=v@self._w+self._a[0]+self._a[1]*gx+self._a[2]*gy
        return float(o[0]),float(o[1])


# ─── GazeMapper ──────────────────────────────────────────────────────────────
class GazeMapper:
    def __init__(self):
        self.Mx=self.My=self.tps=None
        self.kalman=KalmanGaze()
        self.sent_pos=(SCREEN_W//2,SCREEN_H//2)
        self._py=self._pp=None
        self._oefix=OneEuroFilter(0.9,0.04); self._oefiy=OneEuroFilter(0.9,0.04)
        self._oefyw=OneEuroFilter(0.6,0.03); self._oefpt=OneEuroFilter(0.6,0.03)
        self._oefvg=OneEuroFilter(0.9,0.04)
        self._nxa=1.; self._nxb=0.; self._nya=1.; self._nyb=0.

    def calibrate(self, fl, sp):
        Ax=np.array([_feat_x(f[0],f[2]) for f in fl],float)
        Ay=np.array([_feat_y(f[1],f[3]) for f in fl],float)
        bx=np.array([p[0] for p in sp],float)
        by=np.array([p[1] for p in sp],float)
        self.Mx=_ridge(Ax,bx,CAL_RIDGE); self.My=_ridge(Ay,by,CAL_RIDGE)
        px=Ax@self.Mx; py=Ay@self.My
        pxlo,pxhi=px.min(),px.max(); pylo,pyhi=py.min(),py.max()
        self._nxa=(min(SCREEN_W/(pxhi-pxlo),5.) if pxhi>pxlo else 1.)
        self._nxb=-self._nxa*pxlo if pxhi>pxlo else 0.
        self._nya=(min(SCREEN_H/(pyhi-pylo),5.) if pyhi>pylo else 1.)
        self._nyb=-self._nya*pylo if pyhi>pylo else 0.
        try:
            g2d=[(f[2]*TPS_IRIS_WEIGHT+f[0],f[3]*TPS_IRIS_WEIGHT+f[1]) for f in fl]
            self.tps=ThinPlateSpline()
            self.tps.fit(g2d,[p[0] for p in sp],[p[1] for p in sp],lam=1e-3)
        except Exception as e:
            print(f"[TPS hata] {e}"); self.tps=None
        self.kalman=KalmanGaze(); self.sent_pos=(SCREEN_W//2,SCREEN_H//2)
        self._py=self._pp=None
        for f in (self._oefix,self._oefiy,self._oefyw,self._oefpt,self._oefvg): f.reset()

    def map(self, features, blinking=False):
        if self.Mx is None or features is None: return self.kalman.predict_only()
        if blinking: return self.sent_pos
        yaw,pitch,irx,iry,vg=features
        if self._py is not None:
            if abs(yaw-self._py)>HEAD_DELTA_GATE or abs(pitch-self._pp)>HEAD_DELTA_GATE:
                self._py,self._pp=yaw,pitch; return self.sent_pos
        self._py,self._pp=yaw,pitch
        t=time.perf_counter()
        yaw=self._oefyw.update(yaw,t); pitch=self._oefpt.update(pitch,t)
        irx=self._oefix.update(irx,t); iry=self._oefiy.update(iry,t)
        vg=self._oefvg.update(vg,t)
        if self.tps is not None:
            rx,ry=self.tps.predict(irx*TPS_IRIS_WEIGHT+yaw, iry*TPS_IRIS_WEIGHT+pitch)
            sx=float(np.clip(rx+GAZE_X_OFFSET,0,SCREEN_W-1))
            sy=float(np.clip(ry+GAZE_Y_OFFSET,0,SCREEN_H-1))
        else:
            rx=float(_feat_x(yaw,irx)@self.Mx); ry=float(_feat_y(pitch,iry)@self.My)
            sx=float(np.clip(self._nxa*rx+self._nxb+GAZE_X_OFFSET,0,SCREEN_W-1))
            sy=float(np.clip(self._nya*ry+self._nyb+GAZE_Y_OFFSET,0,SCREEN_H-1))
        sm=self.kalman.update((sx,sy))
        spd=math.hypot(self.kalman.x[2,0],self.kalman.x[3,0])
        dead=ADAPTIVE_DEAD_BASE+int(ADAPTIVE_DEAD_BONUS*max(0.,1.-spd/ADAPTIVE_DEAD_SPEED))
        if abs(sm[0]-self.sent_pos[0])<dead and abs(sm[1]-self.sent_pos[1])<dead:
            self.kalman.x[0,0]=float(self.sent_pos[0]); self.kalman.x[2,0]=0.
            self.kalman.x[1,0]=float(self.sent_pos[1]); self.kalman.x[3,0]=0.
            return self.sent_pos
        self.sent_pos=sm; return sm


# ─── Kırpma dedektörü (normal) ───────────────────────────────────────────────
class BlinkDetector:
    def __init__(self):
        self.blink_frames=0; self.in_blink=False
        self.blink_times=collections.deque(maxlen=3)

    def update(self, ear, now):
        double=False
        if ear<EAR_BLINK:
            self.blink_frames+=1; is_bl=self.blink_frames>=2
        else:
            if self.in_blink and self.blink_frames>=2:
                self.blink_times.append(now)
                if len(self.blink_times)>=2 and now-self.blink_times[-2]<=DOUBLE_BLINK_MAX_GAP:
                    double=True; self.blink_times.clear()
            self.blink_frames=0; is_bl=False
        self.in_blink=is_bl; return is_bl, double


# ─── Uzun göz kapama dedektörü (klavye toggle) ───────────────────────────────
class LongBlinkDetector:
    """
    Gözler EAR_BLINK altında DURATION saniye sürekli kapalı kalırsa
    toggle sinyali üretir. Tetiklendikten sonra gözler açılana kadar
    bir daha tetiklenmez.
    """
    def __init__(self, duration: float = 2.0):
        self.duration   = duration
        self._start     = None
        self._triggered = False

    def update(self, ear: float, now: float) -> bool:
        if ear < EAR_BLINK:
            if self._start is None:
                self._start = now
            if not self._triggered and (now - self._start) >= self.duration:
                self._triggered = True
                return True           # → klavyeyi aç/kapat
        else:
            self._start     = None
            self._triggered = False
        return False

    def progress(self, now: float) -> float:
        """0.0–1.0 arası kapatma ilerlemesi (kamera HUD için)."""
        if self._start is None or self._triggered:
            return 0.0
        return min((now - self._start) / self.duration, 1.0)


# ─── Türkçe kelime tahmini ───────────────────────────────────────────────────
_TR_WORDS = sorted(set([
    "bir","iki","üç","dört","beş","altı","yedi","sekiz","dokuz","on",
    "yirmi","otuz","elli","yüz","bin",
    "bu","şu","o","ben","sen","biz","siz","onlar",
    "bana","sana","ona","bizi","benim","senin","bizim",
    "ve","ile","de","da","için","gibi","kadar","sonra","önce",
    "ama","ya","en","çok","var","yok","ne","daha","çünkü",
    "eğer","ancak","hem","veya","belki","hatta","sadece","artık",
    "yine","zaten","hiç","her","nasıl","neden","nerede","kim",
    "şimdi","bugün","yarın","dün","sabah","öğle","akşam","gece",
    "gün","yıl","ay","hafta","saat","dakika","zaman",
    "burada","orada","buraya","oraya","içinde","dışında",
    "üstünde","altında","yanında","karşısında",
    "ev","okul","iş","para","insan","adam","kadın","çocuk",
    "aile","anne","baba","kardeş","arkadaş",
    "doktor","hastane","ilaç","hemşire","ambulans","acil",
    "su","yemek","ekmek","et","meyve","sebze","çay","kahve","süt",
    "kitap","telefon","bilgisayar","araba","yol","şehir","ülke",
    "kapı","pencere","masa","sandalye","yatak","oda","banyo","mutfak",
    "iyi","kötü","güzel","büyük","küçük","yeni","eski",
    "uzun","kısa","sıcak","soğuk","hızlı","yavaş","kolay","zor",
    "doğru","yanlış","hasta","mutlu","üzgün","yorgun","aç","tok",
    "açık","kapalı","temiz","ağır","hafif","sessiz",
    "olmak","gitmek","gelmek","yapmak","söylemek","görmek","bilmek",
    "istemek","vermek","almak","başlamak","bitmek","açmak","kapamak",
    "yemek","içmek","uyumak","kalkmak","oturmak","durmak","yürümek",
    "okumak","yazmak","dinlemek","beklemek","çalışmak","oynamak",
    "konuşmak","anlamak","düşünmek","aramak","bulmak","sevmek",
    "yardım","teşekkür","lütfen","evet","hayır","tamam",
    "merhaba","günaydın","hoşça","üzgünüm","pardon","dikkat",
    "acı","ağrı","sıkıntı","sorun","çözüm","cevap","soru",
    "bilgi","haber","mesaj","randevu","kontrol","ilaç",
    "sağlık","hastane","doktor","ameliyat","tedavi",
    "ac","acik","agri","aile","ak",
]))

def turkish_predict(prefix: str, n: int = 5) -> list:
    if not prefix: return []
    p = prefix.lower()
    matches = [w for w in _TR_WORDS if w.startswith(p) and w != p]
    matches.sort(key=lambda w: len(w))
    return matches[:n]


# ─── Gaze Klavye ─────────────────────────────────────────────────────────────
class GazeKeyboard:
    """
    Türkçe Q sanal klavye.

    • Bakış ile seçim (dwell): DWELL_T saniye aynı tuşa bakılırsa seçilir.
    • Türkçe kelime tahmini: 5 öneri gösterilir, bakışla seçilir.
    • Seçilen karakterler aktif uygulamaya Unicode olarak gönderilir.
    • Gaze koordinatları doğrudan gaze_mapper çıktısından alınır;
      mouse imleci değil, göz bakış noktası kullanılır.
    """

    ROWS = [
        ['Q','W','E','R','T','Y','U','I','O','P','Ğ','Ü','⌫'],
        ['A','S','D','F','G','H','J','K','L','Ş','İ','↵'],
        ['Z','X','C','V','B','N','M','Ö','Ç'],
        ['BOŞLUK', 'SİL'],
    ]

    WIN_NAME  = "Klavye"
    DWELL_T   = 1.0      # saniye — tuş seçim süresi
    COOLDOWN  = 0.45     # saniye — seçim sonrası bekleme (çift seçimi önler)
    FLASH_T   = 0.25     # saniye — seçim flaş animasyonu süresi

    # Canvas boyutları
    WIN_W     = 1280
    WIN_H     = 490
    TITLE_H   = 31       # Windows başlık çubuğu

    # Renk paleti (BGR)
    C_BG      = ( 22,  22,  22)
    C_TEXTBOX = ( 12,  12,  12)
    C_KEY     = ( 55,  55,  55)
    C_HOV     = ( 60, 140,  60)
    C_SEL     = ( 50, 220,  50)
    C_PRED    = ( 45,  70, 110)
    C_PRED_H  = ( 75, 115, 175)
    C_TXT     = (230, 230, 230)
    C_PROG    = ( 60, 220,  60)
    C_GAZE    = ( 50, 220, 220)
    C_BORDER  = ( 80,  80,  80)

    TEXT_H    = 72    # yazılan metin alanı yüksekliği
    PRED_H    = 56    # tahmin satırı yüksekliği
    PAD       = 3

    def __init__(self, win_x: int = 0, win_y: int = None, dwell_t: float = 1.0):
        self.win_x   = win_x
        self.win_y   = win_y if win_y is not None else SCREEN_H - self.WIN_H - self.TITLE_H
        self.DWELL_T = dwell_t
        self.visible = False

        self._typed     = ""           # görünen metin tamponu
        self._preds     = []           # tahmin listesi
        self._hover     = None         # ("key",(r,c)) veya ("pred",i)
        self._dwell_t0  = None
        self._last_sel  = 0.0
        self._flash     = None
        self._flash_t0  = 0.0
        self._gaze_cx   = 0           # gaze canvas koordinatı (çizim için)
        self._gaze_cy   = 0

        self._key_rects  = {}         # {(r,c): (x0,y0,x1,y1)}
        self._pred_rects = {}         # {i: (x0,y0,x1,y1)}
        self._compute_layout()

    # ── Layout ────────────────────────────────────────────────────────────

    def _compute_layout(self):
        self._key_rects.clear(); self._pred_rects.clear()
        W   = self.WIN_W
        p   = self.PAD
        ky0 = self.TEXT_H + self.PRED_H
        n_rows = len(self.ROWS)
        row_h  = (self.WIN_H - ky0) // n_rows

        for r, row in enumerate(self.ROWS):
            y0 = ky0 + r * row_h + p
            y1 = ky0 + (r + 1) * row_h - p

            if row == ['BOŞLUK', 'SİL']:
                sw = int(W * 0.72)
                segs = [(0, sw), (sw, W)]
            else:
                n  = len(row)
                kw = W // n
                segs = [(i*kw, (i+1)*kw if i<n-1 else W) for i in range(n)]

            for c, (x0, x1) in enumerate(segs):
                self._key_rects[(r, c)] = (x0+p, y0, x1-p, y1)

        # Tahmin kutuları
        if self._preds:
            n  = len(self._preds)
            pw = W // n
            for i in range(n):
                x0 = i * pw + p
                x1 = (i+1)*pw - p if i < n-1 else W - p
                self._pred_rects[i] = (x0, self.TEXT_H+p, x1, self.TEXT_H+self.PRED_H-p)

    # ── Pencere yönetimi ──────────────────────────────────────────────────

    def show(self):
        if not self.visible:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN_NAME, self.WIN_W, self.WIN_H)
            cv2.moveWindow(self.WIN_NAME, self.win_x, self.win_y)
            self.visible = True

    def hide(self):
        if self.visible:
            try: cv2.destroyWindow(self.WIN_NAME)
            except: pass
            self.visible = False
            self._hover = None; self._dwell_t0 = None

    def toggle(self):
        self.hide() if self.visible else self.show()

    # ── Koordinat dönüşümü ────────────────────────────────────────────────

    def _to_canvas(self, gx: int, gy: int):
        """Ekran gaze koordinatını canvas koordinatına çevir."""
        return gx - self.win_x, gy - self.win_y - self.TITLE_H

    def _hit_key(self, cx, cy):
        for (r,c),(x0,y0,x1,y1) in self._key_rects.items():
            if x0<=cx<=x1 and y0<=cy<=y1: return (r,c)
        return None

    def _hit_pred(self, cx, cy):
        for i,(x0,y0,x1,y1) in self._pred_rects.items():
            if x0<=cx<=x1 and y0<=cy<=y1: return i
        return None

    # ── Ana güncelleme (her karede çağrılır) ─────────────────────────────

    def update(self, gaze_x: int, gaze_y: int):
        """gaze_x, gaze_y: gaze_mapper'dan gelen ekran koordinatları."""
        if not self.visible: return
        now = time.perf_counter()

        cx, cy = self._to_canvas(gaze_x, gaze_y)
        self._gaze_cx, self._gaze_cy = cx, cy

        # Cooldown
        if now - self._last_sel < self.COOLDOWN: return

        hk   = self._hit_key(cx, cy)
        hp   = self._hit_pred(cx, cy)
        tgt  = ("key", hk) if hk is not None else ("pred", hp) if hp is not None else None

        if tgt == self._hover:
            if tgt is not None and self._dwell_t0 is not None:
                if (now - self._dwell_t0) >= self.DWELL_T:
                    self._select(tgt, now)
        else:
            self._hover   = tgt
            self._dwell_t0 = now if tgt is not None else None

    def _select(self, tgt, now):
        self._last_sel = now
        self._flash    = tgt
        self._flash_t0 = now
        self._hover    = None
        self._dwell_t0 = None

        kind, val = tgt

        if kind == "pred" and val is not None and val < len(self._preds):
            word = self._preds[val]
            # Son yarım kelimeyi sil ve tahmini yaz
            words    = self._typed.split()
            fragment = words[-1] if words else ""
            for _ in fragment: send_vk(VK_BACK)
            for ch in word: send_unicode_char(ch)
            send_unicode_char(' ')
            self._typed = (' '.join(words[:-1]) + (' ' if len(words)>1 else '') + word + ' ').lstrip()
            self._preds = []
            self._compute_layout()
            return

        if kind == "key" and val is not None:
            r, c = val
            key  = self.ROWS[r][c]
            self._type_key(key)

        # Tahminleri güncelle
        words = self._typed.split()
        last  = words[-1] if words else ""
        self._preds = turkish_predict(last.lower())
        self._compute_layout()

    def _type_key(self, key: str):
        if key == '⌫':
            if self._typed: self._typed = self._typed[:-1]
            send_vk(VK_BACK)
        elif key == '↵':
            self._typed += '\n'; send_vk(VK_RETURN)
        elif key == 'BOŞLUK':
            self._typed += ' '; send_unicode_char(' ')
        elif key == 'SİL':
            for _ in self._typed: send_vk(VK_BACK)
            self._typed = ""
        else:
            ch = key.lower()
            self._typed += ch; send_unicode_char(ch)

    def dwell_progress(self) -> float:
        if self._hover and self._dwell_t0:
            return min((time.perf_counter()-self._dwell_t0)/self.DWELL_T, 1.0)
        return 0.0

    # ── Çizim ─────────────────────────────────────────────────────────────

    def draw(self):
        now = time.perf_counter()
        W, H = self.WIN_W, self.WIN_H
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = self.C_BG

        # ── Metin tamponu ──
        canvas[:self.TEXT_H] = self.C_TEXTBOX
        disp = self._typed[-62:] if len(self._typed)>62 else self._typed
        cv2.putText(canvas, disp+'|', (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, self.C_TXT, 2, cv2.LINE_AA)
        cv2.line(canvas, (0, self.TEXT_H-1), (W, self.TEXT_H-1), self.C_BORDER, 1)

        # ── Tahmin çubukları ──
        pred_bg = (35, 55, 85)
        canvas[self.TEXT_H:self.TEXT_H+self.PRED_H] = pred_bg
        for i, (x0,y0,x1,y1) in self._pred_rects.items():
            word   = self._preds[i] if i < len(self._preds) else ""
            is_hov = self._hover == ("pred", i)
            flash  = self._flash == ("pred", i) and (now-self._flash_t0) < self.FLASH_T
            color  = self.C_SEL if flash else (self.C_PRED_H if is_hov else self.C_PRED)
            cv2.rectangle(canvas, (x0,y0), (x1,y1), color, -1)
            cv2.rectangle(canvas, (x0,y0), (x1,y1), (80,110,160), 1)
            if is_hov and self._dwell_t0 and not flash:
                pct = min((now-self._dwell_t0)/self.DWELL_T, 1.0)
                bw  = int((x1-x0)*pct)
                cv2.rectangle(canvas, (x0, y1-5), (x0+bw, y1), self.C_PROG, -1)
            (tw,th),_ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 1)
            cv2.putText(canvas, word, ((x0+x1-tw)//2, (y0+y1+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.C_TXT, 1, cv2.LINE_AA)
        cv2.line(canvas,(0,self.TEXT_H+self.PRED_H-1),(W,self.TEXT_H+self.PRED_H-1),self.C_BORDER,1)

        # ── Tuşlar ──
        for (r,c),(x0,y0,x1,y1) in self._key_rects.items():
            key    = self.ROWS[r][c]
            is_hov = self._hover == ("key", (r,c))
            flash  = self._flash == ("key", (r,c)) and (now-self._flash_t0) < self.FLASH_T
            color  = self.C_SEL if flash else (self.C_HOV if is_hov else self.C_KEY)
            cv2.rectangle(canvas, (x0,y0), (x1,y1), color, -1)
            cv2.rectangle(canvas, (x0,y0), (x1,y1), self.C_BORDER, 1)

            # Dwell yay animasyonu
            if is_hov and self._dwell_t0 and not flash:
                pct  = min((now-self._dwell_t0)/self.DWELL_T, 1.0)
                ck   = ((x0+x1)//2, (y0+y1)//2)
                rk   = min(x1-x0, y1-y0)//2 - 6
                if rk > 4:
                    cv2.ellipse(canvas, ck, (rk,rk), -90, 0, int(360*pct),
                                self.C_PROG, 3, cv2.LINE_AA)

            fsize = 0.52 if len(key)>2 else (0.65 if len(key)==2 else 0.80)
            thick = 1 if len(key)>1 else 2
            (tw,th),_ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fsize, thick)
            cv2.putText(canvas, key, ((x0+x1-tw)//2, (y0+y1+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fsize, self.C_TXT, thick, cv2.LINE_AA)

        # ── Gaze imleci ──
        gcx, gcy = self._gaze_cx, self._gaze_cy
        if 0 <= gcx < W and 0 <= gcy < H:
            cv2.circle(canvas, (gcx, gcy), 12, self.C_GAZE, 2, cv2.LINE_AA)
            cv2.circle(canvas, (gcx, gcy),  3, self.C_GAZE, -1)

        # ── Alt bilgi notu ──
        cv2.putText(canvas,
                    f"Secim: {self.DWELL_T:.1f}sn bakis  |  2sn goz kapat = klavyeyi kapat",
                    (10, H-7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA)
        return canvas


# ─── Kalibrasyon ─────────────────────────────────────────────────────────────
def _trim_mean(samples, trim=0.15):
    arr=np.array(samples,float); n=len(arr); lo=int(n*trim); hi=n-lo
    return [np.mean(np.sort(arr[:,c])[lo:hi]) if hi>lo else np.mean(arr[:,c])
            for c in range(arr.shape[1])]

def _best_samples(samples):
    cut=int(len(samples)*0.45); late=samples[cut:] if len(samples)>12 else samples
    if len(late)<4: return late
    arr=np.array(late,float)
    vel=np.sqrt(np.diff(arr[:,2])**2+np.diff(arr[:,3])**2)
    th=np.percentile(vel,60); keep=[0]+[i+1 for i,v in enumerate(vel) if v<=th]
    s=[late[i] for i in keep]; return s if len(s)>=5 else late

def run_calibration(cap, fl):
    win="KALIBRASYON"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    feat_list=[]; scr_pts=[]; pt_idx=0

    while pt_idx < len(CAL_POINTS_REL):
        rx,ry=CAL_POINTS_REL[pt_idx]
        sx,sy=int(rx*SCREEN_W),int(ry*SCREEN_H)
        t0=time.perf_counter(); samps=[]

        while True:
            frame,result,_,_ = process_frame(cap, fl)
            if frame is None: cv2.destroyWindow(win); return None
            el=time.perf_counter()-t0
            if result.face_landmarks and result.facial_transformation_matrixes and el>CAL_WAIT:
                lms=result.face_landmarks[0]
                if compute_ear(lms)>EAR_BLINK:
                    f=compute_gaze_features(lms,result.facial_transformation_matrixes[0])
                    if f: samps.append(f)

            cf=np.zeros((SCREEN_H,SCREEN_W,3),np.uint8)
            face_ok=bool(result.face_landmarks and result.facial_transformation_matrixes)
            cv2.putText(cf,"YUZ ALGILANDI" if face_ok else "YUZ ALGILANAMADI",
                        (30,SCREEN_H-60),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                        (0,200,0) if face_ok else (0,0,220),2,cv2.LINE_AA)
            msg=f"Noktaya bakin  ({pt_idx+1} / {len(CAL_POINTS_REL)})"
            (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,1.2,2)
            cv2.putText(cf,msg,((SCREEN_W-tw)//2,70),cv2.FONT_HERSHEY_SIMPLEX,1.2,(200,200,200),2,cv2.LINE_AA)
            cv2.putText(cf,"q = iptal",(30,SCREEN_H-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(90,90,90),1,cv2.LINE_AA)
            for i,(frx,fry) in enumerate(CAL_POINTS_REL):
                if i!=pt_idx: cv2.circle(cf,(int(frx*SCREEN_W),int(fry*SCREEN_H)),8,(40,40,40),-1)
            dc=(0,255,255) if int(el*6)%2==0 else (0,120,120)
            if el>CAL_WAIT:
                ap=min((el-CAL_WAIT)/CAL_COLLECT,1.0)
                cv2.ellipse(cf,(sx,sy),(34,34),-90,0,int(360*ap),(0,230,0),4,cv2.LINE_AA)
            cv2.circle(cf,(sx,sy),20,dc,-1); cv2.circle(cf,(sx,sy),20,(255,255,255),2,cv2.LINE_AA)
            if samps: cv2.putText(cf,f"ornek:{len(samps)}",(sx-40,sy+50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,200,100),1,cv2.LINE_AA)
            cv2.imshow(win,cf)
            if cv2.waitKey(1)&0xFF==ord('q'): cv2.destroyWindow(win); return None
            if el>=CAL_WAIT+CAL_COLLECT: break

        if len(samps)>=10:
            feat_list.append(_trim_mean(_best_samples(samps))); scr_pts.append((sx,sy))
        pt_idx+=1

    cv2.destroyWindow(win)
    return (feat_list, scr_pts) if len(feat_list)>=5 else None


# ─── Ana döngü ────────────────────────────────────────────────────────────────
def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    fl = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gm   = GazeMapper()
    bd   = BlinkDetector()
    lbd  = LongBlinkDetector(duration=2.0)    # klavye toggle
    kb   = GazeKeyboard(win_x=0, dwell_t=1.0)
    ftms = collections.deque(maxlen=30)
    mouse_active  = True
    click_enabled = True
    kb_mode       = False     # klavye modunda mouse hareketi dondurulur

    cal = run_calibration(cap, fl)
    if cal is None:
        cap.release(); fl.close(); return
    gm.calibrate(*cal)

    while True:
        frame, result, ow, oh = process_frame(cap, fl)
        if frame is None: break

        ftms.append(time.perf_counter())
        fps = ((len(ftms)-1)/(ftms[-1]-ftms[0])) if len(ftms)>1 else 0.

        blinking = False
        gaze_pos = gm.sent_pos    # varsayılan: son bilinen konum

        if result.face_landmarks and result.facial_transformation_matrixes:
            lms = result.face_landmarks[0]
            left_pts      = [lm_to_px(lms[i],ow,oh) for i in LEFT_EYE]
            right_pts     = [lm_to_px(lms[i],ow,oh) for i in RIGHT_EYE]
            left_iris_pts = [lm_to_px(lms[i],ow,oh) for i in LEFT_IRIS]
            right_iris_pts= [lm_to_px(lms[i],ow,oh) for i in RIGHT_IRIS]
            lx0,ly0,lx1,ly1=eye_bbox(left_pts); rx0,ry0,rx1,ry1=eye_bbox(right_pts)
            cv2.rectangle(frame,(lx0,ly0),(lx1,ly1),(0,220,0),2)
            cv2.rectangle(frame,(rx0,ry0),(rx1,ry1),(0,220,0),2)
            draw_iris(frame,left_iris_pts); draw_iris(frame,right_iris_pts)

            ear = compute_ear(lms)
            now = ftms[-1]
            blinking, double_blink = bd.update(ear, now)

            # ── Uzun göz kapama → klavye toggle ──────────────────────────────
            if lbd.update(ear, now):
                kb_mode = not kb_mode
                if kb_mode: kb.show()
                else:       kb.hide()

            # ── Normal çift kırpma → tıklama (yalnızca klavye kapalıyken) ───
            if double_blink and click_enabled and not kb_mode:
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN,0,0,0,0)
                user32.mouse_event(MOUSEEVENTF_LEFTUP,  0,0,0,0)

            ear_c=(0,0,200) if blinking else (0,200,0)
            cv2.putText(frame,f"EAR:{ear:.2f}",(10,oh-40),cv2.FONT_HERSHEY_SIMPLEX,0.6,ear_c,1,cv2.LINE_AA)

            feats   = compute_gaze_features(lms, result.facial_transformation_matrixes[0])
            gaze_pos = gm.map(feats, blinking=blinking)

            if kb_mode:
                # Klavye modu: gaze → klavye (mouse hareketi yok)
                kb.update(gaze_pos[0], gaze_pos[1])
            else:
                # Normal mod: gaze → mouse
                if gaze_pos and mouse_active:
                    user32.SetCursorPos(gaze_pos[0], gaze_pos[1])

            tps_lbl="TPS" if gm.tps is not None else "POLY"
            if kb_mode:
                st_txt,st_col="KLAVYE",(0,220,220)
            elif double_blink and click_enabled:
                st_txt,st_col="TIKLADI!",(0,220,255)
            elif blinking:
                st_txt,st_col="KIRPMA",(0,80,220)
            elif not mouse_active:
                st_txt,st_col="DURDURULDU",(0,165,255)
            else:
                st_txt,st_col="ACIK",(0,220,0)

            # ── Uzun kırpma ilerleme çubuğu ───────────────────────────────
            lbd_pct = lbd.progress(now)
            if lbd_pct > 0.02:
                bar_w = int(300 * lbd_pct)
                lbl   = "Klavye aciliyor..." if not kb_mode else "Klavye kapaniyor..."
                cv2.rectangle(frame,(10,oh-20),(10+bar_w,oh-8),(0,200,200),-1)
                cv2.putText(frame,lbl,(10,oh-24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,220,220),1,cv2.LINE_AA)

        else:
            gm.map(None)
            st_txt,st_col="YUZ YOK",(0,0,220)
            tps_lbl="--"

        cv2.putText(frame,f"FPS:{fps:.0f}",(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,230,230),2,cv2.LINE_AA)
        cv2.putText(frame,f"GOZ:{st_txt}",(10,64),cv2.FONT_HERSHEY_SIMPLEX,0.85,st_col,2,cv2.LINE_AA)
        cv2.putText(frame,f"MODEL:{tps_lbl}",(10,96),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,60),1,cv2.LINE_AA)
        cl_lbl="ACIK" if click_enabled else "KAPALI"
        cl_col=(0,200,200) if click_enabled else (80,80,80)
        cv2.putText(frame,f"2xKIRPMA TIKLAMA:{cl_lbl}",(10,122),cv2.FONT_HERSHEY_SIMPLEX,0.55,cl_col,1,cv2.LINE_AA)
        kb_lbl="KLAVYE:ACIK" if kb_mode else "2sn goz kapat=klavye"
        cv2.putText(frame,kb_lbl,(10,148),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,220,220) if kb_mode else (120,120,120),1,cv2.LINE_AA)
        cv2.putText(frame,"q:cik  c:kalibrasyon  p:mouse  b:tiklama",
                    (10,oh-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(120,120,120),1,cv2.LINE_AA)

        cv2.imshow("Goz Algilama", frame)

        if kb.visible:
            cv2.imshow(kb.WIN_NAME, kb.draw())

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break
        elif key == ord('c'):
            nc = run_calibration(cap, fl)
            if nc: gm.calibrate(*nc)
        elif key == ord('p'): mouse_active = not mouse_active
        elif key == ord('b'): click_enabled = not click_enabled

    cap.release(); cv2.destroyAllWindows(); fl.close()


if __name__ == "__main__":
    main()