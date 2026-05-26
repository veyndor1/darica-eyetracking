# CLAUDE.md — Göz Takibi İmleç Kontrol Sistemi

## Projeye Başlamadan Önce — Zorunlu Keşif Adımı

Bu proje birden fazla `.py` dosyası içeriyor. **Herhangi bir kod yazmadan önce şunları yap:**

1. Klasördeki tüm `.py` dosyalarını listele.
2. Her `.py` dosyasını baştan sona oku.
3. Hangi sınıf ve fonksiyonun hangi dosyada olduğunu çıkar.
4. Bana kısa bir özet sun: "X dosyasında şunlar var, Y dosyasında şunlar var."
5. Onay verdikten sonra değişikliğe başla.

Bu adımı atlama. Yanlış dosyaya yazılan kod projeyi bozar.

---

## Proje Özeti

MediaPipe FaceLandmarker kullanarak gözün ekranda tam olarak nereye baktığını tespit eden ve fare imlecini o noktaya taşıyan bir Python uygulaması. `face_landmarker.task` modeli proje kökünde.

**Hedef:** Kalibrasyon noktası sayısını artırmadan gaze doğruluğunu en üst seviyeye çıkarmak ve imleç titremesini sıfıra indirmek.

---

## Kritik Bug — İlk Düzeltilecek Şey

`GazeMapper.calibrate()` metodunun sonunda şu iki satır var:

```python
self._nx_a = 1.0;  self._nx_b = 0.0   # ← BUG
self._ny_a = 1.0;  self._ny_b = 0.0   # ← BUG
```

Bunlar az önce hesaplanan normalizasyonu silip etkisiz bırakıyor. **Sadece bu iki satırı sil.** Normalizasyon hesabının kendisine, Kalman resetine, One Euro filter resetine dokunma.

Silme işlemi sonunda `calibrate()` sonu şöyle görünmeli — `_nx_a`/`_nx_b` ile ilgili hiçbir reset satırı olmayacak:

```python
# Bu blok KALACAK (normalizasyon hesabı):
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

# Bu blok KALACAK (diğer resetler — bunlara dokunma):
self.kalman = KalmanGaze()
self.sent_pos = (SCREEN_W // 2, SCREEN_H // 2)
self._prev_yaw = None
self._prev_pitch = None
for f in (...):
    f.reset()

# BUG OLAN SATIRLAR BURADA OLMAYACAK ← silindi
```

---

## Görev Listesi

Görevleri **sırayla** uygula. Bir görev bitmeden sonrakine geçme.

---

### GÖREV 1 — Bug Fix

Yukarıdaki iki `_nx_a / _ny_a` reset satırını sil. Başka hiçbir şeye dokunma. Bitti.

---

### GÖREV 2 — Polinom Özellikler + Ridge Regresyon

**Neden gerekli:** Mevcut model 3 özellikle lineer mapping yapıyor. Göz küresi küresel olduğu için köşelere bakışta lineer model 150–300 px sapıyor.

#### Adım 2a — `_build_feature` fonksiyonu

Mevcut `_feat_x` ve `_feat_y` fonksiyonlarını **kaldır**, yerine tek bir fonksiyon yaz. Bu fonksiyonu diğer yardımcı fonksiyonların yanına koy:

```python
def _build_feature(yaw, pitch, iris_rel_x, iris_rel_y, vergence=0.0):
    """
    12 boyutlu polinom özellik vektörü.
    Cross-term'ler köşe sapmalarını düzeltir.
    vergence: sol_iris_x - sag_iris_x (mesafe bilgisi, varsayılan 0)
    """
    return np.array([
        yaw,
        pitch,
        iris_rel_x,
        iris_rel_y,
        yaw * iris_rel_x,     # yatay çapraz
        pitch * iris_rel_y,   # dikey çapraz
        yaw * iris_rel_y,     # eksen çapraz — baş dönüklüğü Y'yi etkiler
        pitch * iris_rel_x,   # eksen çapraz — baş eğimi X'i etkiler
        yaw ** 2,
        pitch ** 2,
        vergence,
        1.0
    ], dtype=float)
```

#### Adım 2b — Ridge çözücü

`_build_feature`'ın hemen altına ekle:

```python
def _ridge_solve(A, b, lam):
    """Ridge regresyon: (AᵀA + λI)⁻¹ Aᵀb"""
    M = A.T @ A
    M += lam * np.eye(M.shape[0])
    return np.linalg.solve(M, A.T @ b)
```

#### Adım 2c — `calibrate()` güncelle

`lstsq` satırlarını `_ridge_solve` ile değiştir. `Ax` ve `Ay` artık `_build_feature` kullanacak:

```python
# ESKİ:
Ax = np.array([_feat_x(f[0], f[2]) for f in features_list], dtype=float)
Ay = np.array([_feat_y(f[1], f[3]) for f in features_list], dtype=float)
self.Mx, _, _, _ = np.linalg.lstsq(Ax, bx, rcond=None)
self.My, _, _, _ = np.linalg.lstsq(Ay, by, rcond=None)

# YENİ:
Ax = np.array([_build_feature(*f) for f in features_list], dtype=float)
bx = np.array([p[0] for p in screen_pts], dtype=float)
by = np.array([p[1] for p in screen_pts], dtype=float)
self.Mx = _ridge_solve(Ax, bx, CAL_RIDGE)
self.My = _ridge_solve(Ax, by, CAL_RIDGE)
```

#### Adım 2d — `map()` güncelle

`raw_x` / `raw_y` hesabını güncelle:

```python
feat  = _build_feature(yaw, pitch, iris_rel_x, iris_rel_y)
raw_x = float(feat @ self.Mx)
raw_y = float(feat @ self.My)
```

---

### GÖREV 3 — Thin Plate Spline (TPS)

**Neden:** TPS kalibrasyon noktalarında matematiksel olarak sıfır hata verir. 13 noktayla polinom'dan belirgin şekilde daha iyi interpolasyon sağlar. Scipy gerekmez.

#### Adım 3a — `ThinPlateSpline` sınıfı

`GazeMapper` sınıfından **hemen önce** dosyaya ekle:

```python
class ThinPlateSpline:
    """
    2D → 2D Thin Plate Spline. Scipy gerektirmez.
    Kalibrasyon noktalarında hata = 0, arada C² pürüzsüz interpolasyon.
    lam > 0 : yumuşatma (titreme önler), lam = 0 : tam interpolasyon.
    """

    @staticmethod
    def _rbf(r: float) -> float:
        return r * r * math.log(r + 1e-10)

    def fit(self, src_pts, dst_x, dst_y, lam: float = 1e-4):
        """
        src_pts : list[(gx, gy)] — gaze özellik uzayı
        dst_x   : list[float]   — ekran X
        dst_y   : list[float]   — ekran Y
        """
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

        self._w = sol[:N]   # RBF ağırlıkları (N, 2)
        self._a = sol[N:]   # Lineer katsayılar (3, 2)

    def predict(self, gx: float, gy: float):
        """(gx, gy) → (screen_x, screen_y)"""
        rbf_vals = np.array([
            self._rbf(math.hypot(gx - s[0], gy - s[1]))
            for s in self._src
        ])
        out = rbf_vals @ self._w + self._a[0] + self._a[1] * gx + self._a[2] * gy
        return float(out[0]), float(out[1])
```

#### Adım 3b — `GazeMapper.__init__()` güncelle

```python
self._tps = None
```

satırını `__init__` içine ekle.

#### Adım 3c — `_to_gaze2d` yardımcı metodu

`GazeMapper` içine ekle:

```python
def _to_gaze2d(self, yaw, pitch, iris_rel_x, iris_rel_y, vergence=0.0):
    """
    4D özellikleri TPS kaynak uzayına (2D) indir.
    iris sinyali kafa açısından daha kararlı → 0.6 ağırlık.
    vergence büyükse kullanıcı yakın — iris ağırlığını hafif artır.
    """
    w_iris = min(0.6 + 0.08 * abs(vergence), 0.75)
    w_head = 1.0 - w_iris
    gx = w_head * yaw   + w_iris * iris_rel_x
    gy = w_head * pitch  + w_iris * iris_rel_y
    return gx, gy
```

#### Adım 3d — `calibrate()` içine TPS fit ekle

Ridge hesabından sonra, normalizasyon bloğundan önce:

```python
# TPS fit
src_2d = [self._to_gaze2d(*f) for f in features_list]
self._tps = ThinPlateSpline()
self._tps.fit(src_2d,
              bx.tolist(),
              by.tolist(),
              lam=1e-4)
```

#### Adım 3e — `map()` içinde TPS'i birincil kaynak yap

One Euro filtre uygulandıktan sonra `raw_x`/`raw_y` bloğunu şununla değiştir:

```python
if self._tps is not None:
    sx, sy = self._tps.predict(*self._to_gaze2d(yaw, pitch, iris_rel_x, iris_rel_y))
    sx = float(np.clip(sx + GAZE_X_OFFSET, 0, SCREEN_W - 1))
    sy = float(np.clip(sy + GAZE_Y_OFFSET, 0, SCREEN_H - 1))
else:
    # TPS kalibre edilmemişse polinom fallback
    feat  = _build_feature(yaw, pitch, iris_rel_x, iris_rel_y)
    raw_x = float(feat @ self.Mx)
    raw_y = float(feat @ self.My)
    sx = float(np.clip(self._nx_a * raw_x + self._nx_b + GAZE_X_OFFSET, 0, SCREEN_W - 1))
    sy = float(np.clip(self._ny_a * raw_y + self._ny_b + GAZE_Y_OFFSET, 0, SCREEN_H - 1))
```

---

### GÖREV 4 — Vergence Özelliği

**Neden:** Sol ve sağ iris yatay farkı kullanıcının kameraya mesafesini kodlar.

`compute_gaze_features()` fonksiyonunun return satırını güncelle. Mevcut `left_rx`, `right_rx` değerleri zaten hesaplanıyor — sadece şunu ekle:

```python
vergence = (left_rx - right_rx) if (left_rx is not None and right_rx is not None) else 0.0
return [yaw, pitch, iris_rel_x, iris_rel_y, vergence]
```

`_build_feature` zaten `vergence=0.0` parametresini kabul ediyor, bu yüzden `map()` çağrısında sadece 5. argümanı geç:

```python
feat = _build_feature(yaw, pitch, iris_rel_x, iris_rel_y, vergence)
```

`_to_gaze2d` çağrısını da güncelle:

```python
self._to_gaze2d(yaw, pitch, iris_rel_x, iris_rel_y, vergence)
```

---

## İmleç Titremesi — Dokunulmayacak Parametreler

Bu değerleri **asla değiştirme.** Yeni bir titreme görürsen modeli düzelt, bu parametreleri büyütme.

| Bileşen | Değer | Görevi |
|---|---|---|
| `OneEuroFilter` min_cutoff | 0.9 | Durağan bakışta güçlü yumuşatma |
| `OneEuroFilter` beta | 0.04 | Hızlı harekette gecikmeyi önler |
| `VEL_DAMP_THRESHOLD` | 0.6 px/frame | Mikro-hızı sıfırlar |
| `ADAPTIVE_DEAD_BONUS` | 10 px | Durağan bakışta ölü bölge genişler |
| `HEAD_DELTA_GATE` | 0.12 rad/frame | Ani kafa hareketini dondurur |
| `KalmanGaze` process_noise | 55.0 | Titremeden önce hareket tahmini |
| `KalmanGaze` measure_noise | 5500.0 | Ölçüm güvenilirliği |

---

## `map()` İçinde İşlem Sırası — Bozma

```
1. self.Mx is None veya features is None → predict_only() → return
2. blinking == True → predict_only() → return
3. HEAD_DELTA_GATE kontrolü → aşılırsa predict_only() → return
4. _prev_yaw/_prev_pitch güncelle
5. One Euro filtrele: yaw, pitch, iris_rel_x, iris_rel_y
6. TPS.predict() → (sx, sy)  [veya polinom fallback]
7. GAZE_X_OFFSET / GAZE_Y_OFFSET ekle, clip
8. kalman.update((sx, sy)) → smoothed
9. Adaptif ölü bölge hesapla → hareket yoksa sent_pos döndür
10. sent_pos güncelle → return smoothed
```

Bu sıra değişirse filtre zinciri bozulur ve titreme geri döner.

---

## Sık Yapılan Hatalar

- **`calibrate()` sonunda `_nx_a` resetlemek:** Görev 1'i geri almak. Kesinlikle yapma.
- **`ThinPlateSpline.fit()`'i `map()` içinde çağırmak:** `fit` sadece `calibrate()` içinde çağrılır. `map()` sadece `predict()` çağırır.
- **`_build_feature` boyutunu değiştirip `Mx`/`My` boyutunu güncellemeyi unutmak:** `_ridge_solve` boyut uyumsuzluğu verir. Her iki yeri aynı anda güncelle.
- **Polinom ve TPS'i birlikte map() içinde toplamak:** TPS varsa polinom kullanılmaz. `if self._tps is not None` dalı bunu zaten hallediyor — karıştırma.
- **`_to_gaze2d` ağırlıklarını `map()` içinde hesaplamak:** Bu mantık `_to_gaze2d` metodu içinde. Başka yerde tekrarlama.

---

## Test Kriterleri

Her görev bittikten sonra çalıştır ve kontrol et:

| Kriter | Beklenti |
|---|---|
| Kalibrasyon sonrası 4 köşeye bak | İmleç köşelere ulaşmalı (Görev 1 sonrası) |
| Köşe sapması | ≤ 100 px (Görev 2 sonrası), ≤ 40 px (Görev 3 sonrası) |
| Sabit bakış, 2 saniye | İmleç hiç oynamamalı |
| Çift göz kırpma | Sol tık çalışmaya devam etmeli |
| `c` tuşu | Yeniden kalibrasyon çalışmalı |