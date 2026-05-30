<p align="center">
  <img src="docs/assets/banner.svg" alt="Darıca Eye Tracking" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen.svg" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/lisans-MIT-blue.svg" alt="MIT" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Raspberry%20Pi-lightgrey" alt="Platform" />
  <img src="https://img.shields.io/badge/mediapipe-Face%20Landmarker-teal" alt="MediaPipe" />
</p>

---

## Bu proje ne?

Darıca Eye Tracking, sıradan bir webcam ile göz hareketlerinizi takip edip mouse imlecini kontrol eden bir yazılım. Çift kırpmayla tıklama yapabiliyorsunuz, göz kırpmalarıyla çalışan bir ekran klavyesiyle de yazı yazabiliyorsunuz. Webcam ve Python dışında bir şey gerekmiyor.

Fiziksel engelli bireyler bilgisayar kullanırken genelde pahalı göz takip cihazlarına ihtiyaç duyuyor. Bu projeyi bir webcam ile aynı işi yapabilmek için geliştirdim.

### Neler yapıyor?

<table>
<tr>
<td width="50%">

Göz takibi
- İris pozisyonu ve baş açısıyla mouse kontrolü
- Kişiye özel 13 noktalı kalibrasyon
- Bakış yönünü ekran koordinatına çevirmek için TPS + Ridge regresyon
- Titreşimi bastırmak için Kalman ve One Euro filtre
- Yavaş hareketlerde gereksiz oynamayı engelleyen adaptif dead zone

</td>
<td width="50%">

Kırpma ve klavye
- Çift kırpma sol tıklama olarak çalışıyor
- Göz kırpmasıyla harf harf ilerleyen tarama klavyesi
- Yazarken Türkçe kelime tahmini
- Aşağı bakışta yalancı kırpmayı engelleyen adaptif EAR eşiği
- 2 saniye göz kapatınca klavye açılıp kapanıyor

</td>
</tr>
</table>

## Sistem mimarisi

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Sistem mimarisi" width="100%" />
</p>

<details>
<summary>Teknik detaylar</summary>

<br/>

MediaPipe Face Landmarker yüzden 478 nokta çıkartıyor. Bunların içinden iris landmarkları (468-477) ve baş dönüşüm matrisini alıyorum. Ridge regresyon ve TPS interpolasyonuyla birleştirip kalibre edilmiş ekran koordinatlarına çeviriyorum.

Filtreleme üç katmanlı: One Euro Filter düşük gecikmeyle yumuşatma, Kalman Filter gürültü azaltma, Adaptif Dead Zone da yavaş hareketlerde küçük titreşimleri yok sayma. Üçünü bir arada kullanınca imleç makul düzeyde sabit kalıyor.

Kırpma algılama Eye Aspect Ratio (EAR) üzerinden çalışıyor. Eşik sabit değil, sürekli güncelleniyor. Aşağı bakarken göz kapağı doğal olarak biraz kapanıyor; bunu kırpma sanmasın diye iris Y koordinatını da kontrol ediyorum.

</details>

## Kurulum

Python 3.9+, bir webcam ve Windows 10/11 ya da Raspberry Pi OS gerekiyor.

### Windows

```bash
git clone https://github.com/veyndor1/darica-eyetracking.git
cd darica-eyetracking
pip install -r requirements.txt
```

### Raspberry Pi

```bash
git clone https://github.com/veyndor1/darica-eyetracking.git
cd darica-eyetracking
pip install -r requirements-rpi.txt
```

### MediaPipe modeli

`face_landmarker.task` repoda zaten var. Güncellemek isterseniz:

```bash
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Çalıştırma

```bash
# Windows
python gazetracking.py

# Raspberry Pi
python rasbperrypi.py
```

Program açılınca 13 noktalı bir kalibrasyon ekranı çıkıyor. Her noktaya sırayla bakın, hem başınızı hem gözlerinizi noktaya çevirin. Kalibrasyon bitince imleç gözünüze göre hareket etmeye başlıyor.

### Kısayollar

| Tuş | Ne yapıyor |
|:---:|------------|
| `q` | Çıkış |
| `c` | Yeniden kalibrasyon |
| `p` | Mouse kontrolünü aç/kapat |
| `b` | Tıklamayı aç/kapat |

### Tarama klavyesi

Gözlerinizi 2 saniye kapalı tutunca tarama klavyesi açılıyor (aynı şekilde kapanıyor):

| Hareket | Sonuç |
|---------|-------|
| Tek kırpma | Sonraki harfe geç |
| Çift kırpma | Seçili harfi yaz |
| ~1 sn göz kapalı | Sonraki satıra atla |
| ~2 sn göz kapalı | Klavyeyi aç/kapat |

## Raspberry Pi sürümü

`rasbperrypi.py` dosyası Raspberry Pi için. Windows sürümünden farkları:

- Kamerayı Picamera2 API ya da V4L2 backend ile açıyor
- Ekran boyutunu `xrandr` veya `tkinter` ile tespit ediyor
- Mouse ve klavye kontrolü için `pynput` kullanıyor (X11 ortamı gerekli)

Raspberry Pi 4 veya üstünde çalıştırmanızı öneririm, MediaPipe işlemci yiyor.

## Proje yapısı

```
darica-eyetracking/
├── gazetracking.py            # Ana uygulama (Windows)
├── rasbperrypi.py             # Raspberry Pi sürümü
├── face_landmarker.task       # MediaPipe yüz modeli
├── requirements.txt           # Bağımlılıklar (Windows)
├── requirements-rpi.txt       # Bağımlılıklar (Raspberry Pi)
├── docs/assets/               # Logo, banner, diyagramlar
├── CHANGELOG.md               # Sürüm geçmişi
└── LICENSE                    # MIT lisansı
```

## Lisans

[MIT](LICENSE)
