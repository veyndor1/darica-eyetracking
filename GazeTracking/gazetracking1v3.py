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
MODEL_PATH = os.path.join(_SCRIPT_DIR, "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "face_landmarker.task")

user32   = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

CAL_POINTS_REL = [
    (0.50,0.50),
    (0.08,0.08),(0.92,0.08),(0.08,0.92),(0.92,0.92),
    (0.50,0.08),(0.50,0.92),(0.08,0.50),(0.92,0.50),
    (0.29,0.29),(0.71,0.29),(0.29,0.71),(0.71,0.71),
]
CAL_WAIT=1.2; CAL_COLLECT=1.8; CAL_RIDGE=0.02; EAR_BLINK=0.40
DEAD_ZONE=0; GAZE_X_OFFSET=0; GAZE_Y_OFFSET=-2; GAZE_Y_FLIP=1
HEAD_DELTA_GATE=0.22; TPS_IRIS_WEIGHT=1.5
CAL_EXPAND_X = 1.0
CAL_EXPAND_Y = 1.15
VEL_DAMP_THRESHOLD=0.6; ADAPTIVE_DEAD_BASE=0; ADAPTIVE_DEAD_BONUS=10; ADAPTIVE_DEAD_SPEED=14.0
MOUSEEVENTF_LEFTDOWN=0x0002; MOUSEEVENTF_LEFTUP=0x0004
IRI_DOWN_GATE=0.25

# ─── Unicode tuş gönderici ───────────────────────────────────────────────────
INPUT_KEYBOARD=1; KEYEVENTF_UNICODE=0x0004; KEYEVENTF_KEYUP=0x0002
VK_BACK=0x08; VK_RETURN=0x0D

class _KBI(ctypes.Structure):
    _fields_=[("wVk",ctypes.c_ushort),("wScan",ctypes.c_ushort),
              ("dwFlags",ctypes.c_ulong),("time",ctypes.c_ulong),("dwExtraInfo",ctypes.c_uint64)]
class _INP(ctypes.Structure):
    _fields_=[("type",ctypes.c_ulong),("ki",_KBI),("_p",ctypes.c_ubyte*8)]

def send_char(ch):
    c=ord(ch)
    for fl in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE|KEYEVENTF_KEYUP):
        i=_INP(); i.type=INPUT_KEYBOARD; i.ki.wScan=c; i.ki.dwFlags=fl
        ctypes.windll.user32.SendInput(1,ctypes.byref(i),ctypes.sizeof(_INP))

def send_vk(vk):
    ctypes.windll.user32.keybd_event(vk,0,0,0)
    ctypes.windll.user32.keybd_event(vk,0,KEYEVENTF_KEYUP,0)

# ─── One Euro Filter ─────────────────────────────────────────────────────────
class OEF:
    def __init__(self,mc=0.9,b=0.04,dc=1.0):
        self.mc=mc;self.b=b;self.dc=dc;self._x=None;self._dx=0.;self._t=None
    @staticmethod
    def _a(c,dt): return 1./(1.+1./(2.*math.pi*c*dt))
    def update(self,x,t=None):
        if t is None: t=time.perf_counter()
        if self._x is None: self._x,self._t=float(x),t; return self._x
        dt=max(t-self._t,1e-6); dx=(x-self._x)/dt
        ad=self._a(self.dc,dt); self._dx=ad*dx+(1.-ad)*self._dx
        a=self._a(self.mc+self.b*abs(self._dx),dt)
        self._x=a*float(x)+(1.-a)*self._x; self._t=t; return self._x
    def reset(self): self._x=None;self._dx=0.;self._t=None

# ─── Yardımcı ────────────────────────────────────────────────────────────────
def lm_to_px(lm,w,h): return (int(lm.x*w),int(lm.y*h))

def eye_bbox(pts,px=0.25,py=0.40):
    xs=[p[0] for p in pts];ys=[p[1] for p in pts]
    x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
    pw=int((x1-x0)*px);ph=int((y1-y0)*py)
    return x0-pw,y0-ph,x1+pw,y1+ph

def draw_iris(f,px):
    cx,cy=px[0]; r=max(int(np.mean([np.hypot(p[0]-cx,p[1]-cy) for p in px[1:]])) if len(px)>1 else 8,2)
    cv2.circle(f,(cx,cy),r,(0,0,220),2); cv2.circle(f,(cx,cy),2,(255,230,0),-1)

def compute_ear(lms):
    def _e(h1,h2,v1,v2,v3,v4):
        ho=math.hypot(lms[h1].x-lms[h2].x,lms[h1].y-lms[h2].y)
        return (math.hypot(lms[v1].x-lms[v2].x,lms[v1].y-lms[v2].y)+
                math.hypot(lms[v3].x-lms[v4].x,lms[v3].y-lms[v4].y))/(2.*ho) if ho>1e-6 else 1.
    return (_e(33,133,159,145,158,153)+_e(263,362,386,374,385,380))/2.

def get_head_angles(tm):
    rv,_=cv2.Rodrigues(np.array(tm.data,float).reshape(4,4)[:3,:3])
    return float(rv[1,0]),float(rv[0,0])

def compute_gaze_features(lms,tm):
    yaw,pitch=get_head_angles(tm)
    def im(idx): n=len(idx); return sum(lms[i].x for i in idx)/n,sum(lms[i].y for i in idx)/n
    def es(idx):
        xs=[lms[i].x for i in idx];ys=[lms[i].y for i in idx]
        return min(xs),max(xs),min(ys),max(ys)
    lx0,lx1,ly0,ly1=es(LEFT_EYE);lix,liy=im(LEFT_IRIS);lew=lx1-lx0;leh=ly1-ly0
    rx0,rx1,ry0,ry1=es(RIGHT_EYE);rix,riy=im(RIGHT_IRIS);rew=rx1-rx0;reh=ry1-ry0
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
    return [yaw,pitch,irx,iry*GAZE_Y_FLIP,(lrx-rrx) if (lrx and rrx) else 0.]

def process_frame(cap,fl):
    ret,frame=cap.read()
    if not ret: return None,None,0,0
    h,w=frame.shape[:2]; sc=PROCESS_WIDTH/w if w>PROCESS_WIDTH else 1.
    small=cv2.resize(frame,(int(w*sc),int(h*sc))) if sc<1. else frame
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
    return frame,fl.detect(mp_img),w,h

# ─── Kalman ───────────────────────────────────────────────────────────────────
class KalmanGaze:
    def __init__(self,pn=20.,mn=1400.):
        dt=1/30.; self.x=np.array([[SCREEN_W/2.],[SCREEN_H/2.],[0.],[0.]])
        self.P=np.eye(4)*2000.
        self.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
        self.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        self.Q=np.eye(4)*pn; self.R=np.eye(2)*mn
    def update(self,m):
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
        z=np.array([[float(m[0])],[float(m[1])]]); y=z-self.H@self.x
        S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y; self.P=(np.eye(4)-K@self.H)@self.P; self._d(); return self._c()
    def predict_only(self):
        self.x[2,0]=0.; self.x[3,0]=0.; self.x=self.F@self.x
        self.P=self.F@self.P@self.F.T+self.Q; self._d(); return self._c()
    def _d(self):
        if abs(self.x[2,0])<VEL_DAMP_THRESHOLD: self.x[2,0]=0.
        if abs(self.x[3,0])<VEL_DAMP_THRESHOLD: self.x[3,0]=0.
    def _c(self):
        return (int(np.clip(self.x[0,0],0,SCREEN_W-1)),int(np.clip(self.x[1,0],0,SCREEN_H-1)))

def _fx(y,ix): return np.array([y,ix,y*ix,y**2,ix**2,1.],float)
def _fy(p,iy): return np.array([p,iy,p*iy,p**2,iy**2,1.],float)
def _rdg(A,b,l):
    M=A.T@A; M+=l*np.eye(M.shape[0]); return np.linalg.solve(M,A.T@b)

class TPS:
    @staticmethod
    def _r(r): return r*r*math.log(r+1e-10)
    def fit(self,s,dx,dy,lam=1e-4):
        self._s=np.array(s,float); N=len(self._s)
        K=np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                r=math.hypot(self._s[i,0]-self._s[j,0],self._s[i,1]-self._s[j,1])
                K[i,j]=K[j,i]=self._r(r)
        P=np.hstack([np.ones((N,1)),self._s])
        M=np.block([[K+lam*np.eye(N),P],[P.T,np.zeros((3,3))]])
        rx=np.concatenate([np.array(dx,float),[0,0,0]])
        ry=np.concatenate([np.array(dy,float),[0,0,0]])
        sol=np.linalg.solve(M,np.column_stack([rx,ry]))
        self._w=sol[:N]; self._a=sol[N:]
    def predict(self,gx,gy):
        v=np.array([self._r(math.hypot(gx-s[0],gy-s[1])) for s in self._s])
        o=v@self._w+self._a[0]+self._a[1]*gx+self._a[2]*gy
        return float(o[0]),float(o[1])

class GazeMapper:
    def __init__(self):
        self.Mx=self.My=self.tps=None; self.kalman=KalmanGaze()
        self.sent_pos=(SCREEN_W//2,SCREEN_H//2); self._py=self._pp=None
        self._ox=OEF(0.9,0.04);self._oy=OEF(0.9,0.04)
        self._ow=OEF(0.6,0.03);self._op=OEF(0.6,0.03);self._ov=OEF(0.9,0.04)
        self._nxa=1.;self._nxb=0.;self._nya=1.;self._nyb=0.
    def calibrate(self,fl,sp):
        Ax=np.array([_fx(f[0],f[2]) for f in fl],float)
        Ay=np.array([_fy(f[1],f[3]) for f in fl],float)
        bx=np.array([p[0] for p in sp],float); by=np.array([p[1] for p in sp],float)
        self.Mx=_rdg(Ax,bx,CAL_RIDGE); self.My=_rdg(Ay,by,CAL_RIDGE)
        px=Ax@self.Mx;py=Ay@self.My
        pxl,pxh=px.min(),px.max(); pyl,pyh=py.min(),py.max()
        self._nxa=min(SCREEN_W/(pxh-pxl),5.) if pxh>pxl else 1.
        self._nxb=-self._nxa*pxl if pxh>pxl else 0.
        self._nya=min(SCREEN_H/(pyh-pyl),5.) if pyh>pyl else 1.
        self._nyb=-self._nya*pyl if pyh>pyl else 0.
        try:
            g2d=[(f[2]*TPS_IRIS_WEIGHT+f[0],f[3]*TPS_IRIS_WEIGHT+f[1]) for f in fl]
            self.tps=TPS(); self.tps.fit(g2d,[p[0] for p in sp],[p[1] for p in sp],lam=1e-3)
        except Exception as e:
            print(f"[TPS hata] {e}"); self.tps=None
        self.kalman=KalmanGaze(); self.sent_pos=(SCREEN_W//2,SCREEN_H//2)
        self._py=self._pp=None
        for f in (self._ox,self._oy,self._ow,self._op,self._ov): f.reset()
    def map(self,features,blinking=False):
        if self.Mx is None or features is None: return self.kalman.predict_only()
        if blinking: return self.sent_pos
        yaw,pitch,irx,iry,vg=features
        if self._py is not None:
            if abs(yaw-self._py)>HEAD_DELTA_GATE or abs(pitch-self._pp)>HEAD_DELTA_GATE:
                self._py,self._pp=yaw,pitch; return self.sent_pos
        self._py,self._pp=yaw,pitch
        t=time.perf_counter()
        yaw=self._ow.update(yaw,t);pitch=self._op.update(pitch,t)
        irx=self._ox.update(irx,t);iry=self._oy.update(iry,t);vg=self._ov.update(vg,t)
        if self.tps:
            rx,ry=self.tps.predict(irx*TPS_IRIS_WEIGHT+yaw,iry*TPS_IRIS_WEIGHT+pitch)
        else:
            rx=float(_fx(yaw,irx)@self.Mx);ry=float(_fy(pitch,iry)@self.My)
            rx=self._nxa*rx+self._nxb; ry=self._nya*ry+self._nyb
        rx = (rx - SCREEN_W/2) * CAL_EXPAND_X + SCREEN_W/2
        ry = (ry - SCREEN_H/2) * CAL_EXPAND_Y + SCREEN_H/2
        sx = float(np.clip(rx + GAZE_X_OFFSET, 0, SCREEN_W-1))
        sy = float(np.clip(ry + GAZE_Y_OFFSET, 0, SCREEN_H-1))
        sm=self.kalman.update((sx,sy))
        spd=math.hypot(self.kalman.x[2,0],self.kalman.x[3,0])
        dead=ADAPTIVE_DEAD_BASE+int(ADAPTIVE_DEAD_BONUS*max(0.,1.-spd/ADAPTIVE_DEAD_SPEED))
        if abs(sm[0]-self.sent_pos[0])<dead and abs(sm[1]-self.sent_pos[1])<dead:
            self.kalman.x[0,0]=float(self.sent_pos[0]);self.kalman.x[2,0]=0.
            self.kalman.x[1,0]=float(self.sent_pos[1]);self.kalman.x[3,0]=0.
            return self.sent_pos
        self.sent_pos=sm; return sm

# ─── Normal kırpma (mouse tıklama için) ──────────────────────────────────────
def _is_eye_closed(ear, baseline, prev_ear, in_blink, iry=0.0):
    threshold = baseline * 0.78
    if in_blink:
        # Devam eden kırpmayı iry ile kesme — iris blink sırasında oynayabilir
        return ear < threshold
    # Yeni kırpma başlangıcı: aşağı bakış ise bastır
    if iry > IRI_DOWN_GATE:
        return False
    return (ear < threshold) and (prev_ear - ear > 0.025)

class BlinkDetector:
    DOUBLE_WIN_S = 0.70   # ilk kırpma bittikten sonra ikincisini bekleme süresi

    def __init__(self):
        self.frames=0; self.in_b=False
        self._base=None; self._pe=None
        self._pending=False; self._pending_t=0.0

    def update(self, ear, now, iry=0.0):
        if self._base is None: self._base=ear; self._pe=ear
        closed = _is_eye_closed(ear, self._base, self._pe, self.in_b, iry)
        db = False
        if closed:
            self.frames += 1
        else:
            if self.in_b and self.frames >= 2:
                if self._pending:
                    self._pending = False; db = True
                else:
                    self._pending = True; self._pending_t = now
            self.frames = 0
            self._base = 0.97*self._base + 0.03*ear
            # Timeout: blink-end kontrolünden SONRA, sadece göz açıkken
            if self._pending and (now - self._pending_t) >= self.DOUBLE_WIN_S:
                self._pending = False
        self._pe = ear
        self.in_b = self.frames >= 2
        return self.in_b, db

# ─────────────────────────────────────────────────────────────────────────────
#  TARAMA KIRPMA DEDEKTÖRÜ
#
#  Olaylar:
#    'advance'  — tek kırpma  → sonraki harfe geç
#    'double'   — çift kırpma → mevcut harfi seç
#    'row_skip' — 1sn tut     → sonraki satıra atla (göz açılınca)
#    'toggle'   — 2sn tut     → klavyeyi aç/kapat (gözler kapalıyken)
#
#  Yanlış pozitif koruması:
#    • _is_eye_closed() adaptif eşik + ani düşüş kriteri kullanır
#    • MIN_FRAMES: en az 2 ardışık kare kapalı kalmalı
#    • NOISE_S: 70ms'den kısa kapanmalar yok sayılır
#
#  Çift kırpma mantığı (bekleme penceresi):
#    İlk kırpma biter → 380ms pencere başlar, henüz 'advance' gönderilmez
#    380ms içinde ikinci kırpma → 'double' gönderilir
#    380ms geçer, ikinci kırpma gelmezse → 'advance' gönderilir
# ─────────────────────────────────────────────────────────────────────────────
class ScanBlinkDetector:
    NOISE_S      = 0.07    # bu kadardan kısa kapanmalar gürültü sayılır
    ROW_SKIP_S   = 0.85    # ≥ bu kadar tut → 'row_skip' (göz açılınca)
    TOGGLE_S     = 2.00    # ≥ bu kadar tut → 'toggle' (kapalıyken tetikler)
    DOUBLE_WIN_S = 0.55    # iki kırpma arası bu kadar ise → 'double'
    MIN_FRAMES   = 2       # gerçek kırpma için minimum ardışık kapalı kare

    def __init__(self):
        self._t0      = None    # gözler kapandığı an
        self._frames  = 0       # ardışık kapalı kare sayısı
        self._fired   = False   # toggle zaten tetiklendi mi
        self._base    = None    # adaptif EAR taban çizgisi (_is_eye_closed ile uyumlu)
        self._pe      = None    # bir önceki frame EAR
        # Pending single blink (çift kırpma tespiti için)
        self._pending   = False
        self._pending_t = 0.0

    @property
    def is_pending(self):
        """Dışarıdan okunabilir pending durumu — klavye görsel için."""
        return self._pending

    def update(self, ear: float, now: float, iry: float = 0.0):
        """
        Döner: (event, hold_progress)
          event         : None | 'advance' | 'double' | 'row_skip' | 'toggle'
          hold_progress : 0.0–1.0  (ROW_SKIP_S eşiğine ne kadar yaklaşıldı)
        """
        if self._base is None:
            self._base = ear; self._pe = ear

        in_blink = self._t0 is not None
        closed   = _is_eye_closed(ear, self._base, self._pe, in_blink, iry)

        if closed:
            # ── GÖZ KAPALI ───────────────────────────────────────────
            self._frames += 1
            if self._t0 is None:
                self._t0 = now; self._fired = False
            held = now - self._t0
            self._pe = ear

            # Toggle: gözler hâlâ kapalıyken 2sn'de tetiklenir
            if not self._fired and held >= self.TOGGLE_S:
                self._fired   = True
                self._pending = False
                return 'toggle', 1.0

            # Göz kapalıyken pending timeout ASLA ateşlenmez —
            # ikinci kırpma henüz sürebilir, timeout onu 'advance'e dönüştürür.
            return None, min(held / self.ROW_SKIP_S, 1.0)

        else:
            # ── GÖZ AÇIK ─────────────────────────────────────────────
            self._base = 0.97 * self._base + 0.03 * ear
            self._pe   = ear

            prev_frames  = self._frames
            self._frames = 0

            if self._t0 is not None:
                held        = now - self._t0
                self._t0    = None
                was_toggled = self._fired
                self._fired = False

                if not was_toggled and prev_frames >= self.MIN_FRAMES and held >= self.NOISE_S:
                    if held >= self.ROW_SKIP_S:
                        self._pending = False
                        return 'row_skip', 0.0
                    else:
                        if self._pending:
                            # İkinci kırpma: pending aktifken göz açıldı → ÇİFT KIRPMA
                            # Timeout süresi geçmiş olsa bile blink-end önceliklidir.
                            self._pending = False
                            return 'double', 0.0
                        else:
                            self._pending   = True
                            self._pending_t = now

            # Blink bitmedi ya da bitti ama pending sürüyorsa → timeout kontrol
            # (Bu kontrol her zaman blink-end mantığından SONRA gelir)
            if self._pending and (now - self._pending_t) >= self.DOUBLE_WIN_S:
                self._pending = False
                return 'advance', 0.0
            return None, 0.0


# ─── Türkçe kelime tahmini ───────────────────────────────────────────────────
_WORDS = sorted(set([
    "bir","iki","üç","dört","beş","altı","yedi","sekiz","dokuz","on",
    "yirmi","otuz","elli","yüz","bin","ve","ile","de","da","için",
    "gibi","kadar","sonra","önce","ama","ya","en","çok","var","yok",
    "ne","daha","çünkü","eğer","ancak","hem","veya","belki","hatta",
    "sadece","artık","yine","zaten","hiç","her","nasıl","neden",
    "nerede","kim","şimdi","bugün","yarın","dün","sabah","öğle",
    "akşam","gece","gün","yıl","ay","hafta","saat","zaman",
    "burada","orada","buraya","içinde","dışında","üstünde","altında",
    "yanında","ev","okul","iş","para","insan","adam","kadın","çocuk",
    "aile","anne","baba","kardeş","arkadaş","doktor","hastane","ilaç",
    "hemşire","acil","su","yemek","ekmek","çay","kahve","kitap",
    "telefon","bilgisayar","araba","yol","şehir","ülke","kapı",
    "masa","yatak","oda","banyo","mutfak","iyi","kötü","güzel",
    "büyük","küçük","yeni","eski","sıcak","soğuk","kolay","zor",
    "doğru","yanlış","hasta","mutlu","üzgün","yorgun","aç","tok",
    "olmak","gitmek","gelmek","yapmak","söylemek","görmek","bilmek",
    "istemek","vermek","almak","başlamak","bitmek","yemek","içmek",
    "uyumak","kalkmak","oturmak","yürümek","okumak","yazmak",
    "dinlemek","beklemek","çalışmak","konuşmak","anlamak","aramak",
    "bulmak","sevmek","yardım","teşekkür","lütfen","evet","hayır",
    "tamam","merhaba","günaydın","üzgünüm","pardon","dikkat",
    "acı","ağrı","sorun","çözüm","bilgi","mesaj","randevu",
    "sağlık","tedavi","ameliyat","ben","sen","o","biz","siz","onlar",
    "bana","sana","ona","benim","senin","bizim","bu","şu",
]))

def tr_predict(prefix: str, n: int = 6) -> list:
    if not prefix: return []
    p = prefix.lower()
    m = [w for w in _WORDS if w.startswith(p) and w != p]
    return sorted(m, key=len)[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  TARAMA KLAVYESİ
#
#  Tarama doğrusal ilerler: Tahminler (varsa) → Satır0 → Satır1 → Satır2 → Satır3 → (başa)
#  Her satır içinde harften harfe geçilir; satır sonu gelindiğinde bir sonraki satıra atlanır.
#
#  Kırpma olayları:
#    'advance'  → bir sonraki harfe geç
#    'double'   → mevcut harfi/tahmini seç
#    'row_skip' → bir sonraki satırın başına atla
# ─────────────────────────────────────────────────────────────────────────────
class ScanKeyboard:

    ROWS = [
        ['Q','W','E','R','T','Y','U','I','O','P','Ğ','Ü'],
        ['A','S','D','F','G','H','J','K','L','Ş','İ'],
        ['Z','X','C','V','B','N','M','Ö','Ç'],
        ['BOŞLUK','⌫','↵','SİL'],
    ]

    WIN_NAME = "Tarama Klavye"
    WIN_W    = 1280
    WIN_H    = 500
    TITLE_H  = 31

    # ── Renkler (BGR) ────────────────────────────────────────────────
    C_BG       = ( 18,  18,  18)
    C_TXTBOX   = (  8,   8,   8)
    C_KEY      = ( 50,  50,  50)
    C_CUR      = ( 30, 210,  30)    # şu anki tuş — parlak yeşil
    C_PENDING  = (  0, 200, 220)    # çift kırpma bekleniyor — sarımsı
    C_CUR_ROW  = ( 22,  55,  22)    # şu anki satırın diğer tuşları
    C_PRED_BG  = ( 38,  55,  90)
    C_PRED_CUR = ( 75, 125, 200)
    C_FLASH    = (220, 240, 100)    # seçim flaşı
    C_PROG_G   = ( 40, 200,  40)    # ilerleme çubuğu (yeşil — row_skip'e doğru)
    C_PROG_C   = ( 40, 200, 200)    # ilerleme çubuğu (camgöbeği — toggle'a doğru)
    C_BORDER   = ( 65,  65,  65)
    C_TXT      = (230, 230, 230)
    C_DIM_TXT  = (140, 140, 140)
    C_STATUS   = ( 85,  85,  85)

    def __init__(self, win_x=0, win_y=None):
        self.win_x  = win_x
        self.win_y  = win_y if win_y is not None else SCREEN_H - self.WIN_H - self.TITLE_H
        self.visible = False

        self.typed   = ""
        self.preds   = []

        # Doğrusal tarama pozisyonu
        self._srow   = 0    # scan row index (tahmin satırı dahil)
        self._scol   = 0    # scan column index

        # Görsel durumlar
        self._pending = False
        self._flash   = False
        self._flash_t = 0.0
        self._hold_p  = 0.0
        self._last_type_t = 0.0   # send_char sonrası waitKey bastırma için

    # ── Tüm tarama satırlarını üret ───────────────────────────────────
    def _display_rows(self):
        """Mevcut tahminler dahil tüm satırları döndürür."""
        rows = []
        if self.preds:
            rows.append(('pred', list(self.preds)))
        for r in self.ROWS:
            rows.append(('key', list(r)))
        return rows

    def _current_item(self):
        """(kind, label) — şu an üzerindeki öğe."""
        rows = self._display_rows()
        if not rows:
            return 'key', '?'
        self._clamp(rows)
        kind, items = rows[self._srow]
        return kind, items[self._scol]

    def _clamp(self, rows):
        """Satır/sütun indekslerini geçerli aralıkta tut."""
        if not rows:
            self._srow = 0; self._scol = 0; return
        self._srow = self._srow % len(rows)
        self._scol = self._scol % len(rows[self._srow][1])

    # ── Pencere ───────────────────────────────────────────────────────
    def show(self):
        if not self.visible:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN_NAME, self.WIN_W, self.WIN_H)
            cv2.moveWindow(self.WIN_NAME, self.win_x, self.win_y)
            self.visible = True
            self._srow = 0; self._scol = 0

    def hide(self):
        if self.visible:
            try: cv2.destroyWindow(self.WIN_NAME)
            except: pass
            self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()

    # ── Olay işleyici ─────────────────────────────────────────────────
    def on_event(self, event: str, hold_progress: float = 0.0, pending: bool = False):
        self._hold_p  = hold_progress
        self._pending = pending

        if event == 'advance':
            self._advance()
        elif event == 'double':
            self._select()
            self._flash = True; self._flash_t = time.perf_counter()
        elif event == 'row_skip':
            self._row_skip()

    def _advance(self):
        rows = self._display_rows()
        if not rows: return
        self._clamp(rows)
        _, items = rows[self._srow]
        self._scol += 1
        if self._scol >= len(items):
            self._scol = 0
            self._srow = (self._srow + 1) % len(rows)

    def _row_skip(self):
        rows = self._display_rows()
        if not rows: return
        self._srow = (self._srow + 1) % len(rows)
        self._scol = 0

    def _select(self):
        kind, label = self._current_item()

        if kind == 'pred':
            words = self.typed.split()
            frag  = words[-1] if words else ""
            for _ in frag: send_vk(VK_BACK)
            for ch in label: send_char(ch)
            send_char(' ')
            self._last_type_t = time.perf_counter()
            self.typed = (' '.join(words[:-1]) + (' ' if len(words)>1 else '') + label + ' ').lstrip()
            self.preds = []
        else:
            self._type_key(label)
            words = self.typed.split()
            last  = words[-1].lower() if words else ""
            self.preds = tr_predict(last)

        # Seçimden sonra başa dön
        self._srow = 0; self._scol = 0

    def _type_key(self, key: str):
        if key == '⌫':
            if self.typed: self.typed = self.typed[:-1]
            send_vk(VK_BACK)
        elif key == '↵':
            self.typed += '\n'; send_vk(VK_RETURN)
        elif key == 'BOŞLUK':
            self.typed += ' '; send_char(' ')
        elif key == 'SİL':
            for _ in self.typed: send_vk(VK_BACK)
            self.typed = ""; self.preds = []
        else:
            ch = key.lower(); self.typed += ch; send_char(ch)
            self._last_type_t = time.perf_counter()

    # ── Çizim ─────────────────────────────────────────────────────────
    def draw(self):
        now = time.perf_counter()
        W, H = self.WIN_W, self.WIN_H
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = self.C_BG
        PAD = 3

        rows = self._display_rows()
        self._clamp(rows)
        cur_kind, cur_label = self._current_item()
        flash_on = self._flash and (now - self._flash_t) < 0.30

        # ── İlerleme çubuğu (en üst, 12px) ──────────────────────────
        if self._hold_p > 0.02:
            bw = int((W - 2) * min(self._hold_p, 1.0))
            c  = self.C_PROG_C if self._hold_p >= 1.0 else self.C_PROG_G
            cv2.rectangle(canvas, (1, 1), (1 + bw, 11), c, -1)

        # ── Kısa ipucu metni ─────────────────────────────────────────
        if self._pending:
            hint = ">> TEKRAR KIRP = SEÇ  |  Bekle = İlerle  |  1sn tut = Satır atla"
            hcol = self.C_PENDING
        else:
            hint = "Kırp=İlerle  |  2x Kırp=Seç  |  1sn tut=Satır atla  |  2sn tut=Kapat"
            hcol = self.C_STATUS
        cv2.putText(canvas, hint, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, hcol, 1, cv2.LINE_AA)

        # ── Yazı alanı (65px) ─────────────────────────────────────────
        TXT_Y0 = 30
        TXT_H  = 62
        canvas[TXT_Y0:TXT_Y0+TXT_H] = self.C_TXTBOX
        disp = self.typed[-65:] if len(self.typed) > 65 else self.typed
        cv2.putText(canvas, disp+'|', (10, TXT_Y0+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.92, self.C_TXT, 2, cv2.LINE_AA)
        cv2.line(canvas, (0, TXT_Y0+TXT_H-1), (W, TXT_Y0+TXT_H-1), self.C_BORDER, 1)

        # ── Büyük mevcut tuş göstergesi (72px) ───────────────────────
        BIG_Y0 = TXT_Y0 + TXT_H
        BIG_H  = 72
        canvas[BIG_Y0:BIG_Y0+BIG_H] = (12, 12, 12)

        # Renk: pending→ camgöbeği, flash→ sarı, normal→ yeşil
        if flash_on:          big_c = self.C_FLASH
        elif self._pending:   big_c = self.C_PENDING
        else:                 big_c = self.C_CUR

        label_map = {'BOŞLUK':'[ BOŞLUK ]','SİL':'[ SİL ]','⌫':'[ ⌫ ]','↵':'[ ENTER ]'}
        big_lbl = label_map.get(cur_label, cur_label)

        (tw, th), _ = cv2.getTextSize(big_lbl, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.putText(canvas, big_lbl, ((W-tw)//2, BIG_Y0 + BIG_H//2 + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, big_c, 3, cv2.LINE_AA)

        # Satır / sütun konum bilgisi
        pos_lbl = f"Satır {self._srow+1}/{len(rows)}  •  Sütun {self._scol+1}"
        cv2.putText(canvas, pos_lbl, (W-230, BIG_Y0+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.C_STATUS, 1, cv2.LINE_AA)

        cv2.line(canvas, (0, BIG_Y0+BIG_H-1), (W, BIG_Y0+BIG_H-1), self.C_BORDER, 1)

        # ── Klavye grid ───────────────────────────────────────────────
        KEY_Y0 = BIG_Y0 + BIG_H
        avail  = H - KEY_Y0
        n_rows = len(rows)
        row_h  = avail // n_rows if n_rows else avail

        for ri, (kind, items) in enumerate(rows):
            y0 = KEY_Y0 + ri * row_h + PAD
            y1 = KEY_Y0 + (ri + 1) * row_h - PAD
            n  = len(items)
            kw = W // n
            is_cur_row = (ri == self._srow)

            for ci, key in enumerate(items):
                x0 = ci * kw + PAD
                x1 = (ci+1)*kw - PAD if ci < n-1 else W - PAD
                is_cur = is_cur_row and (ci == self._scol)

                # Arka plan rengi
                if is_cur:
                    if flash_on:        fc = self.C_FLASH
                    elif self._pending: fc = self.C_PENDING
                    else:               fc = self.C_PRED_CUR if kind=='pred' else self.C_CUR
                elif is_cur_row:
                    fc = self.C_PRED_BG if kind=='pred' else self.C_CUR_ROW
                else:
                    fc = self.C_PRED_BG if kind=='pred' else self.C_KEY

                cv2.rectangle(canvas, (x0,y0), (x1,y1), fc, -1)
                cv2.rectangle(canvas, (x0,y0), (x1,y1), self.C_BORDER, 1)

                # Şu anki satırın kenarlarını belirginleştir
                if is_cur_row and not is_cur:
                    cv2.rectangle(canvas, (x0,y0), (x1,y1), (60,120,60) if kind=='key' else (60,90,140), 1)

                fsize = 0.50 if len(key)>3 else (0.62 if len(key)>1 else 0.80)
                thick = 1 if len(key)>1 else 2
                tc    = (15,15,15) if is_cur and not flash_on else self.C_TXT
                (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fsize, thick)
                cv2.putText(canvas, key, ((x0+x1-tw)//2, (y0+y1+th)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, fsize, tc, thick, cv2.LINE_AA)

        # ── Sol kenar satır göstergesi ────────────────────────────────
        for ri in range(n_rows):
            cy = KEY_Y0 + ri*row_h + row_h//2
            if ri == self._srow:
                cv2.circle(canvas, (7, cy), 6, self.C_CUR, -1)
            else:
                cv2.circle(canvas, (7, cy), 3, (50,50,50), -1)

        return canvas


# ─── Kalibrasyon ─────────────────────────────────────────────────────────────
def _tm(s,tr=0.15):
    a=np.array(s,float);n=len(a);lo=int(n*tr);hi=n-lo
    return [np.mean(np.sort(a[:,c])[lo:hi]) if hi>lo else np.mean(a[:,c]) for c in range(a.shape[1])]

def _bs(s):
    cut=int(len(s)*0.45);late=s[cut:] if len(s)>12 else s
    if len(late)<4: return late
    a=np.array(late,float); v=np.sqrt(np.diff(a[:,2])**2+np.diff(a[:,3])**2)
    th=np.percentile(v,60); keep=[0]+[i+1 for i,vv in enumerate(v) if vv<=th]
    r=[late[i] for i in keep]; return r if len(r)>=5 else late

def run_calibration(cap, fl):
    win="KALIBRASYON"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    flist=[]; spts=[]; idx=0
    while idx<len(CAL_POINTS_REL):
        rx,ry=CAL_POINTS_REL[idx]; sx,sy=int(rx*SCREEN_W),int(ry*SCREEN_H)
        t0=time.perf_counter(); samps=[]
        while True:
            frame,result,_,_=process_frame(cap,fl)
            if frame is None: cv2.destroyWindow(win); return None
            el=time.perf_counter()-t0
            if result.face_landmarks and result.facial_transformation_matrixes and el>CAL_WAIT:
                lms=result.face_landmarks[0]
                if compute_ear(lms)>EAR_BLINK:
                    f=compute_gaze_features(lms,result.facial_transformation_matrixes[0])
                    if f: samps.append(f)
            cf=np.zeros((SCREEN_H,SCREEN_W,3),np.uint8)
            fok=bool(result.face_landmarks and result.facial_transformation_matrixes)
            cv2.putText(cf,"YUZ ALGILANDI" if fok else "YUZ ALGILANAMADI",(30,SCREEN_H-60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,0) if fok else (0,0,220),2,cv2.LINE_AA)
            msg=f"Noktaya bakin ({idx+1}/{len(CAL_POINTS_REL)})"
            (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,1.2,2)
            cv2.putText(cf,msg,((SCREEN_W-tw)//2,70),cv2.FONT_HERSHEY_SIMPLEX,1.2,(200,200,200),2,cv2.LINE_AA)
            cv2.putText(cf,"q=iptal",(30,SCREEN_H-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(90,90,90),1,cv2.LINE_AA)
            for i,(frx,fry) in enumerate(CAL_POINTS_REL):
                if i!=idx: cv2.circle(cf,(int(frx*SCREEN_W),int(fry*SCREEN_H)),8,(40,40,40),-1)
            dc=(0,255,255) if int(el*6)%2==0 else (0,120,120)
            if el>CAL_WAIT:
                ap=min((el-CAL_WAIT)/CAL_COLLECT,1.)
                cv2.ellipse(cf,(sx,sy),(34,34),-90,0,int(360*ap),(0,230,0),4,cv2.LINE_AA)
            cv2.circle(cf,(sx,sy),20,dc,-1); cv2.circle(cf,(sx,sy),20,(255,255,255),2,cv2.LINE_AA)
            if samps: cv2.putText(cf,f"ornek:{len(samps)}",(sx-40,sy+50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,200,100),1,cv2.LINE_AA)
            cv2.imshow(win,cf)
            if cv2.waitKey(1)&0xFF==ord('q'): cv2.destroyWindow(win); return None
            if el>=CAL_WAIT+CAL_COLLECT: break
        if len(samps)>=10: flist.append(_tm(_bs(samps))); spts.append((sx,sy))
        idx+=1
    cv2.destroyWindow(win)
    return (flist,spts) if len(flist)>=5 else None

# ─── Ana döngü ────────────────────────────────────────────────────────────────
def main():
    options=FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True)
    fl=FaceLandmarker.create_from_options(options)
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)

    gm   = GazeMapper()
    bd   = BlinkDetector()       # çift kırpma → mouse tıklama
    sbd  = ScanBlinkDetector()   # tarama klavye olayları
    kb   = ScanKeyboard(win_x=0)
    ftms = collections.deque(maxlen=30)
    mouse_active  = True
    click_enabled = True
    kb_mode       = False

    cal=run_calibration(cap,fl)
    if cal is None: cap.release(); fl.close(); return
    gm.calibrate(*cal)

    while True:
        frame,result,ow,oh=process_frame(cap,fl)
        if frame is None: break
        ftms.append(time.perf_counter())
        fps=((len(ftms)-1)/(ftms[-1]-ftms[0])) if len(ftms)>1 else 0.

        blinking=False; gaze_pos=gm.sent_pos

        if result.face_landmarks and result.facial_transformation_matrixes:
            lms=result.face_landmarks[0]
            lp=[lm_to_px(lms[i],ow,oh) for i in LEFT_EYE]
            rp=[lm_to_px(lms[i],ow,oh) for i in RIGHT_EYE]
            lip=[lm_to_px(lms[i],ow,oh) for i in LEFT_IRIS]
            rip=[lm_to_px(lms[i],ow,oh) for i in RIGHT_IRIS]
            lx0,ly0,lx1,ly1=eye_bbox(lp); rx0,ry0,rx1,ry1=eye_bbox(rp)
            cv2.rectangle(frame,(lx0,ly0),(lx1,ly1),(0,220,0),2)
            cv2.rectangle(frame,(rx0,ry0),(rx1,ry1),(0,220,0),2)
            draw_iris(frame,lip); draw_iris(frame,rip)

            ear=compute_ear(lms); now=ftms[-1]
            feats=compute_gaze_features(lms,result.facial_transformation_matrixes[0])
            iry = feats[3] if feats is not None else 0.0
            blinking,double_blink=bd.update(ear,now,iry)

            # ── Tarama kırpma dedektörü ──────────────────────────────
            scan_event, hold_p = sbd.update(ear, now, iry)

            # ── Klavye toggle (her modda) ────────────────────────────
            if scan_event == 'toggle':
                kb_mode = not kb_mode
                kb.toggle()

            if kb_mode:
                # ── KLAVYE MODU ──────────────────────────────────────
                if scan_event in ('advance', 'double', 'row_skip'):
                    kb.on_event(scan_event, hold_p, sbd.is_pending)
                else:
                    kb.on_event(None, hold_p, sbd.is_pending)
            else:
                # ── NORMAL MOUSE MODU ────────────────────────────────
                if double_blink and click_enabled:
                    user32.mouse_event(MOUSEEVENTF_LEFTDOWN,0,0,0,0)
                    user32.mouse_event(MOUSEEVENTF_LEFTUP,  0,0,0,0)

            ear_c=(0,0,200) if blinking else (0,200,0)
            cv2.putText(frame,f"EAR:{ear:.2f}",(10,oh-40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,ear_c,1,cv2.LINE_AA)

            gaze_pos=gm.map(feats,blinking=blinking)

            if not kb_mode and gaze_pos and mouse_active:
                user32.SetCursorPos(gaze_pos[0],gaze_pos[1])

            # ── Blink tutma göstergesi kamerada ─────────────────────
            if hold_p > 0.05:
                bw=int(250*hold_p)
                lbl="1sn TAMAM → Satır atlıyor" if hold_p>=1. else f"Tut → Satır Atla ({hold_p*100:.0f}%)"
                cv2.rectangle(frame,(10,oh-22),(10+bw,oh-10),(50,220,50),-1)
                cv2.putText(frame,lbl,(10,oh-25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(50,250,50),1,cv2.LINE_AA)

            tps_lbl="TPS" if gm.tps else "POLY"
            if kb_mode:
                _, cur_lbl = kb._current_item()
                st_txt=f"KLAVYE [{cur_lbl[:5]}]"; st_col=(0,220,220)
            elif double_blink and click_enabled:
                st_txt,st_col="TIKLADI!",(0,220,255)
            elif blinking:
                st_txt,st_col="KIRPMA",(0,80,220)
            else:
                st_txt,st_col="ACIK",(0,220,0)
        else:
            gm.map(None); st_txt,st_col="YUZ YOK",(0,0,220); tps_lbl="--"

        cv2.putText(frame,f"FPS:{fps:.0f}",(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,230,230),2,cv2.LINE_AA)
        cv2.putText(frame,f"GOZ:{st_txt}",(10,64),cv2.FONT_HERSHEY_SIMPLEX,0.85,st_col,2,cv2.LINE_AA)
        cv2.putText(frame,f"MODEL:{tps_lbl}",(10,96),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,60),1,cv2.LINE_AA)
        kb_hint="[KLAVYE ACIK] 2sn kapat=kapat" if kb_mode else "2sn goz kapat=klavye ac"
        cv2.putText(frame,kb_hint,(10,122),cv2.FONT_HERSHEY_SIMPLEX,0.50,
                    (0,220,220) if kb_mode else (100,100,100),1,cv2.LINE_AA)
        cv2.putText(frame,"q:cik  c:kalibrasyon  p:mouse  b:tiklama",
                    (10,oh-55),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,100),1,cv2.LINE_AA)

        cv2.imshow("Goz Algilama",frame)
        if kb.visible:
            cv2.imshow(kb.WIN_NAME,kb.draw())

        key=cv2.waitKey(1)&0xFF
        # Klavye bir karakter gönderdikten sonra 150ms, o karakterin waitKey'e sızmasını engelle
        if kb_mode and (time.perf_counter() - kb._last_type_t) < 0.15:
            key = 0xFF
        if key==27 or key==ord('q'): break
        elif key==ord('c') and not kb_mode:
            nc=run_calibration(cap,fl)
            if nc: gm.calibrate(*nc)
        elif key==ord('p') and not kb_mode: mouse_active=not mouse_active
        elif key==ord('b') and not kb_mode: click_enabled=not click_enabled

    cap.release(); cv2.destroyAllWindows(); fl.close()

if __name__=="__main__":
    main()