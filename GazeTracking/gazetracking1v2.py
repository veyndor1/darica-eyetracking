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

CAL_POINTS_REL = [
    (0.50,0.50),
    (0.08,0.08),(0.92,0.08),(0.08,0.92),(0.92,0.92),
    (0.50,0.08),(0.50,0.92),(0.08,0.50),(0.92,0.50),
    (0.29,0.29),(0.71,0.29),(0.29,0.71),(0.71,0.71),
]
CAL_WAIT=1.2; CAL_COLLECT=1.8; CAL_RIDGE=0.02; EAR_BLINK=0.40
DEAD_ZONE=0; GAZE_X_OFFSET=0; GAZE_Y_OFFSET=-2; GAZE_Y_FLIP=1
HEAD_DELTA_GATE=0.22; TPS_IRIS_WEIGHT=1.5
VEL_DAMP_THRESHOLD=0.6; ADAPTIVE_DEAD_BASE=0; ADAPTIVE_DEAD_BONUS=5; ADAPTIVE_DEAD_SPEED=12.0
MOUSEEVENTF_LEFTDOWN=0x0002; MOUSEEVENTF_LEFTUP=0x0004
DOUBLE_BLINK_MAX_GAP=1.5

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
    def __init__(self,pn=20.,mn=600.):
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
            sx=float(np.clip(rx+GAZE_X_OFFSET,0,SCREEN_W-1))
            sy=float(np.clip(ry+GAZE_Y_OFFSET,0,SCREEN_H-1))
        else:
            rx=float(_fx(yaw,irx)@self.Mx);ry=float(_fy(pitch,iry)@self.My)
            sx=float(np.clip(self._nxa*rx+self._nxb+GAZE_X_OFFSET,0,SCREEN_W-1))
            sy=float(np.clip(self._nya*ry+self._nyb+GAZE_Y_OFFSET,0,SCREEN_H-1))
        sm=self.kalman.update((sx,sy))
        spd=math.hypot(self.kalman.x[2,0],self.kalman.x[3,0])
        dead=ADAPTIVE_DEAD_BASE+int(ADAPTIVE_DEAD_BONUS*max(0.,1.-spd/ADAPTIVE_DEAD_SPEED))
        if abs(sm[0]-self.sent_pos[0])<dead and abs(sm[1]-self.sent_pos[1])<dead:
            self.kalman.x[0,0]=float(self.sent_pos[0]);self.kalman.x[2,0]=0.
            self.kalman.x[1,0]=float(self.sent_pos[1]);self.kalman.x[3,0]=0.
            return self.sent_pos
        self.sent_pos=sm; return sm

# ─── Normal kırpma (mouse tıklama için) ──────────────────────────────────────
class BlinkDetector:
    def __init__(self):
        self.frames=0;self.in_b=False;self.times=collections.deque(maxlen=3)
    def update(self,ear,now):
        db=False
        if ear<EAR_BLINK: self.frames+=1; isb=self.frames>=2
        else:
            if self.in_b and self.frames>=2:
                self.times.append(now)
                if len(self.times)>=2 and now-self.times[-2]<=DOUBLE_BLINK_MAX_GAP:
                    db=True; self.times.clear()
            self.frames=0; isb=False
        self.in_b=isb; return isb,db

# ─────────────────────────────────────────────────────────────────────────────
#  TARAMA KIRPMA DEDEKTÖRÜ
#  Tek kırpma (<0.55sn)  → 'advance'   : bir adım ilerle
#  Uzun kırpma (≥0.55sn) → 'select'    : seçim yap
#  Çok uzun (≥2.00sn)    → 'toggle'    : klavyeyi aç/kapat
# ─────────────────────────────────────────────────────────────────────────────
class ScanBlinkDetector:
    NOISE   = 0.07    # gürültü eşiği (sn)
    SELECT  = 0.55    # uzun kırpma eşiği (sn) — "0.6sn" hedefine yakın
    TOGGLE  = 2.00    # klavye toggle eşiği (sn)

    def __init__(self):
        self._t0 = None
        self._fired = False   # toggle gözler kapıyken tetiklendi mi

    def update(self, ear: float, now: float):
        """
        Döner: (event, hold_progress)
          event         : None | 'advance' | 'select' | 'toggle'
          hold_progress : 0.0–1.0  → SELECT eşiğine ne kadar yaklaşıldı
                          (klavye üzerindeki ilerleme çubuğu için)
        """
        if ear < EAR_BLINK:                          # ── gözler kapalı ──
            if self._t0 is None:
                self._t0 = now; self._fired = False
            held = now - self._t0
            if not self._fired and held >= self.TOGGLE:
                self._fired = True
                return 'toggle', 1.0                 # gözler hâlâ kapalı
            prog = min(held / self.SELECT, 1.0)
            return None, prog

        else:                                         # ── gözler açıldı ──
            if self._t0 is None:
                return None, 0.0
            held = now - self._t0
            self._t0 = None
            if self._fired:                          # toggle zaten tetiklendi
                self._fired = False
                return None, 0.0
            if held < self.NOISE:                    # çok kısa → gürültü
                return None, 0.0
            if held >= self.SELECT:
                return 'select', 0.0
            return 'advance', 0.0

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
#  Durum makinesi:
#    STATE_ROW  → satırlar sırayla vurgulanır
#    STATE_COL  → seçili satırdaki tuşlar sırayla vurgulanır
#    STATE_PRED → tahmin sözcükleri sırayla vurgulanır
#
#  Kırpma olayları:
#    'advance' → bir sonraki öğeye geç
#    'select'  → şu anki öğeyi seç / bir alt seviyeye in
#    'toggle'  → klavyeyi kapat
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

    # ── Durum sabitleri ──────────────────────────────────────────────
    S_ROW  = 0
    S_COL  = 1
    S_PRED = 2

    # ── Renkler (BGR) ────────────────────────────────────────────────
    C_BG        = ( 18,  18,  18)
    C_TEXTBOX   = (  8,   8,   8)
    C_KEY       = ( 50,  50,  50)
    C_ROW_HL    = ( 30,  90,  30)   # vurgulanan satır zemini
    C_COL_DIM   = ( 20,  70,  20)   # seçili satır (sütun tarama)
    C_COL_HL    = ( 40, 210,  40)   # vurgulanan sütun
    C_PRED_BG   = ( 40,  60, 100)
    C_PRED_HL   = ( 80, 130, 200)
    C_GERI      = ( 30,  80, 130)   # ← GERİ tuşu
    C_GERI_HL   = ( 60, 140, 210)
    C_FLASH     = (200, 230,  80)
    C_TXT       = (235, 235, 235)
    C_TXT_DARK  = ( 30,  30,  30)
    C_PROG      = ( 50, 220,  50)
    C_PROG_SEL  = ( 50, 200, 220)
    C_BORDER    = ( 70,  70,  70)
    C_STATUS    = ( 90,  90,  90)

    def __init__(self, win_x=0, win_y=None):
        self.win_x = win_x
        self.win_y = win_y if win_y is not None else SCREEN_H - self.WIN_H - self.TITLE_H
        self.visible = False

        self.state    = self.S_ROW
        self.row_idx  = 0     # hangi satır (preds dahil)
        self.col_idx  = 0     # 0 = ← GERİ, 1..N = tuşlar
        self.pred_idx = 0     # 0 = ← GERİ, 1..N = tahminler

        self.typed    = ""
        self.preds    = []

        self._flash   = False
        self._flash_t = 0.0
        self._hold_p  = 0.0   # ilerleme çubuğu için blink tutma yüzdesi

    # ── Pencere ───────────────────────────────────────────────────────
    def show(self):
        if not self.visible:
            cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN_NAME, self.WIN_W, self.WIN_H)
            cv2.moveWindow(self.WIN_NAME, self.win_x, self.win_y)
            self.visible = True
            # Klavye açılınca başa dön
            self.state=self.S_ROW; self.row_idx=0
            self.col_idx=0; self.pred_idx=0

    def hide(self):
        if self.visible:
            try: cv2.destroyWindow(self.WIN_NAME)
            except: pass
            self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()

    # ── Olay işleyici ─────────────────────────────────────────────────
    def on_event(self, event: str, hold_progress: float = 0.0):
        """'advance' veya 'select' olayını işle."""
        self._hold_p = hold_progress
        if event == 'advance':
            self._advance()
        elif event == 'select':
            self._select()
            self._flash = True
            self._flash_t = time.perf_counter()

    def _total_rows(self):
        """Kaç satır taranacak (tahmin satırı dahil)."""
        return len(self.ROWS) + (1 if self.preds else 0)

    def _is_pred_row(self):
        """Şu anki satır tahmin satırı mı?"""
        return self.preds and self.row_idx == 0

    def _actual_row(self):
        """Tahmin satırı hariç klavye satır indeksi."""
        return self.row_idx - (1 if self.preds else 0)

    def _advance(self):
        if self.state == self.S_ROW:
            self.row_idx = (self.row_idx + 1) % self._total_rows()

        elif self.state == self.S_COL:
            row = self.ROWS[self._actual_row()]
            # 0 = ← GERİ, 1..n = tuşlar
            self.col_idx = (self.col_idx + 1) % (len(row) + 1)

        elif self.state == self.S_PRED:
            # 0 = ← GERİ, 1..n = tahminler
            self.pred_idx = (self.pred_idx + 1) % (len(self.preds) + 1)

    def _select(self):
        if self.state == self.S_ROW:
            if self._is_pred_row():
                self.state    = self.S_PRED
                self.pred_idx = 0
            else:
                self.state   = self.S_COL
                self.col_idx = 0

        elif self.state == self.S_COL:
            if self.col_idx == 0:           # ← GERİ
                self.state = self.S_ROW
            else:
                key = self.ROWS[self._actual_row()][self.col_idx - 1]
                self._type_key(key)
                # Tahminleri güncelle
                words = self.typed.split()
                last  = words[-1].lower() if words else ""
                self.preds = tr_predict(last)
                self.state   = self.S_ROW
                self.row_idx = 0

        elif self.state == self.S_PRED:
            if self.pred_idx == 0:          # ← GERİ
                self.state = self.S_ROW
            else:
                word  = self.preds[self.pred_idx - 1]
                words = self.typed.split()
                frag  = words[-1] if words else ""
                for _ in frag: send_vk(VK_BACK)
                for ch in word: send_char(ch)
                send_char(' ')
                self.typed = (' '.join(words[:-1]) + (' ' if len(words)>1 else '') + word + ' ').lstrip()
                self.preds    = []
                self.state    = self.S_ROW
                self.row_idx  = 0

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
            ch = key.lower()
            self.typed += ch; send_char(ch)

    # ── Çizim ─────────────────────────────────────────────────────────
    def draw(self):
        now = time.perf_counter()
        W, H = self.WIN_W, self.WIN_H
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = self.C_BG

        PAD   = 3
        TEXT_H = 70
        STATUS_H = 28

        # ── Durum çubuğu (üst) ──────────────────────────────────────
        st_map = {self.S_ROW:"SATIR TARAMA",
                  self.S_COL:"SÜTUN TARAMA",
                  self.S_PRED:"TAHMİN TARAMA"}
        hint_map = {self.S_ROW:"Kısa kırp=İlerle  |  Uzun kırp(≥0.55sn)=Seç  |  2sn=Kapat",
                    self.S_COL:"Kısa kırp=İlerle  |  Uzun kırp=Seç  |  ← GERİ=Satıra dön",
                    self.S_PRED:"Kısa kırp=İlerle  |  Uzun kırp=Kelimeyi seç  |  ← GERİ=Geri"}
        cv2.putText(canvas, f"[{st_map.get(self.state,'?')}]  {hint_map.get(self.state,'')}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_STATUS, 1, cv2.LINE_AA)

        # ── Blink tutma ilerleme çubuğu ─────────────────────────────
        if self._hold_p > 0.02:
            bw = int((W - 20) * self._hold_p)
            c  = self.C_PROG_SEL if self._hold_p >= 1.0 else self.C_PROG
            cv2.rectangle(canvas, (10, STATUS_H - 8), (10 + bw, STATUS_H - 3), c, -1)

        # ── Yazı alanı ──────────────────────────────────────────────
        ty0 = STATUS_H
        canvas[ty0:ty0+TEXT_H] = self.C_TEXTBOX
        disp = self.typed[-65:] if len(self.typed) > 65 else self.typed
        cv2.putText(canvas, disp + '|', (12, ty0+46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, self.C_TXT, 2, cv2.LINE_AA)
        cv2.line(canvas, (0, ty0+TEXT_H-1), (W, ty0+TEXT_H-1), self.C_BORDER, 1)

        # ── Anahtar alanı başlangıcı ─────────────────────────────────
        key_y0   = ty0 + TEXT_H
        n_display = (1 if self.preds else 0) + len(self.ROWS)  # tahmin + 4 satır
        row_h    = (H - key_y0) // n_display

        display_rows = []
        if self.preds:
            display_rows.append(('pred', self.preds))
        for r, row in enumerate(self.ROWS):
            display_rows.append(('key', r))

        flash_on = self._flash and (now - self._flash_t) < 0.28

        for di, (kind, val) in enumerate(display_rows):
            y0 = key_y0 + di * row_h + PAD
            y1 = key_y0 + (di + 1) * row_h - PAD

            # Hangi satır bu tarama dizisinde?
            is_cur_row = (di == self.row_idx)

            if kind == 'pred':
                # ── Tahmin satırı ──────────────────────────────────
                n = len(val) + 1  # +1 için ← GERİ
                pw = W // n
                items = ['← GERİ'] + val

                row_bg = self.C_PRED_HL if (is_cur_row and self.state == self.S_ROW) else \
                         self.C_COL_DIM if (is_cur_row and self.state == self.S_PRED) else \
                         self.C_PRED_BG
                canvas[y0:y1, :] = row_bg

                for ci, item in enumerate(items):
                    x0i = ci * pw + PAD
                    x1i = (ci + 1) * pw - PAD if ci < n - 1 else W - PAD
                    is_hi = (is_cur_row and self.state == self.S_PRED and ci == self.pred_idx)
                    fc = self.C_FLASH if (flash_on and is_hi) else \
                         (self.C_GERI_HL if (is_hi and ci == 0) else
                          self.C_PRED_HL if is_hi else self.C_PRED_BG)
                    if is_hi or ci == 0:
                        cv2.rectangle(canvas, (x0i, y0), (x1i, y1), fc, -1)
                    cv2.rectangle(canvas, (x0i, y0), (x1i, y1), (70, 100, 150), 1)
                    (tw, th), _ = cv2.getTextSize(item, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                    cv2.putText(canvas, item, ((x0i+x1i-tw)//2, (y0+y1+th)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.C_TXT, 1, cv2.LINE_AA)

            else:
                # ── Klavye satırı ──────────────────────────────────
                r = val
                row = self.ROWS[r]

                # Arka plan rengi duruma göre
                if is_cur_row and self.state == self.S_ROW:
                    row_bg = self.C_ROW_HL           # tüm satır vurgulu (satır tarama)
                elif is_cur_row and self.state == self.S_COL:
                    row_bg = self.C_COL_DIM          # satır seçili, sütun taranıyor
                else:
                    row_bg = self.C_BG
                canvas[y0:y1, :] = row_bg

                # Satır başına ← GERİ (sadece sütun taramada, bu satır seçiliyse)
                items = (['← GERİ'] + row) if (is_cur_row and self.state == self.S_COL) else row
                n  = len(items)
                kw = W // n

                for ci, key in enumerate(items):
                    x0k = ci * kw + PAD
                    x1k = (ci + 1) * kw - PAD if ci < n - 1 else W - PAD

                    # Bu sütun vurgulu mu?
                    is_hi_col = (is_cur_row and self.state == self.S_COL and ci == self.col_idx)
                    is_geri   = (key == '← GERİ')

                    fc = self.C_FLASH if (flash_on and is_hi_col) else \
                         (self.C_GERI_HL if (is_hi_col and is_geri) else
                          self.C_COL_HL if is_hi_col else
                          self.C_GERI if is_geri else
                          self.C_KEY)
                    cv2.rectangle(canvas, (x0k, y0), (x1k, y1), fc, -1)
                    cv2.rectangle(canvas, (x0k, y0), (x1k, y1), self.C_BORDER, 1)

                    # Satır tarama sırasında tüm satır vurguysa kenarları belirginleştir
                    if is_cur_row and self.state == self.S_ROW:
                        cv2.rectangle(canvas, (x0k, y0), (x1k, y1), (80,180,80), 1)

                    fsize = 0.52 if len(key) > 3 else (0.62 if len(key) > 1 else 0.82)
                    thick = 1 if len(key) > 1 else 2
                    tc    = self.C_TXT_DARK if fc == self.C_FLASH else self.C_TXT
                    (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fsize, thick)
                    cv2.putText(canvas, key, ((x0k+x1k-tw)//2, (y0+y1+th)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, fsize, tc, thick, cv2.LINE_AA)

        # ── Sol kenar: tarama pozisyon göstergesi ────────────────────
        for di in range(len(display_rows)):
            y0 = key_y0 + di * row_h
            y1 = key_y0 + (di + 1) * row_h
            cy = (y0 + y1) // 2
            if di == self.row_idx:
                cv2.circle(canvas, (8, cy), 6, self.C_PROG, -1)
            else:
                cv2.circle(canvas, (8, cy), 3, self.C_BORDER, -1)

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
            blinking,double_blink=bd.update(ear,now)

            # ── Tarama kırpma dedektörü ──────────────────────────────
            scan_event, hold_p = sbd.update(ear, now)

            # ── Klavye toggle (her modda) ────────────────────────────
            if scan_event == 'toggle':
                kb_mode = not kb_mode
                kb.toggle()

            if kb_mode:
                # ── KLAVYE MODU ──────────────────────────────────────
                if scan_event in ('advance', 'select'):
                    kb.on_event(scan_event, hold_p)
                elif scan_event is None:
                    kb.on_event(None, hold_p)   # ilerleme çubuğu güncelle
            else:
                # ── NORMAL MOUSE MODU ────────────────────────────────
                if double_blink and click_enabled:
                    user32.mouse_event(MOUSEEVENTF_LEFTDOWN,0,0,0,0)
                    user32.mouse_event(MOUSEEVENTF_LEFTUP,  0,0,0,0)

            ear_c=(0,0,200) if blinking else (0,200,0)
            cv2.putText(frame,f"EAR:{ear:.2f}",(10,oh-40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,ear_c,1,cv2.LINE_AA)

            feats=compute_gaze_features(lms,result.facial_transformation_matrixes[0])
            gaze_pos=gm.map(feats,blinking=blinking)

            if not kb_mode and gaze_pos and mouse_active:
                user32.SetCursorPos(gaze_pos[0],gaze_pos[1])

            # ── Blink tutma göstergesi kamerada ─────────────────────
            if hold_p > 0.05:
                bw=int(250*hold_p)
                lbl="Uzun kırpma — SEÇİYOR..." if hold_p>=1. else f"Tut → Seç ({'%.0f'%(hold_p*100)}%)"
                cv2.rectangle(frame,(10,oh-22),(10+bw,oh-10),(50,220,50),-1)
                cv2.putText(frame,lbl,(10,oh-25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(50,250,50),1,cv2.LINE_AA)

            tps_lbl="TPS" if gm.tps else "POLY"
            if kb_mode:
                sc_map={ScanKeyboard.S_ROW:"SATIR",ScanKeyboard.S_COL:"SÜTUN",ScanKeyboard.S_PRED:"TAHMİN"}
                st_txt=f"KLAVYE [{sc_map.get(kb.state,'?')}]"; st_col=(0,220,220)
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
        if key==27 or key==ord('q'): break
        elif key==ord('c'):
            nc=run_calibration(cap,fl)
            if nc: gm.calibrate(*nc)
        elif key==ord('p'): mouse_active=not mouse_active
        elif key==ord('b'): click_enabled=not click_enabled

    cap.release(); cv2.destroyAllWindows(); fl.close()

if __name__=="__main__":
    main()