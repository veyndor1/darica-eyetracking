<p align="center">
  <img src="docs/assets/banner.svg" alt="Darica Eye Tracking" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/veyndor1/darica-eyetracking/actions/workflows/ci.yml"><img src="https://github.com/veyndor1/darica-eyetracking/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/lisans-MIT-blue.svg" alt="MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-brightgreen.svg" alt="Python 3.9+" /></a>
  <a href="https://img.shields.io/badge/platform-Windows%20%7C%20Raspberry%20Pi-lightgrey"><img src="https://img.shields.io/badge/platform-Windows%20%7C%20Raspberry%20Pi-lightgrey" alt="Platform" /></a>
  <a href="https://github.com/veyndor1/darica-eyetracking/stargazers"><img src="https://img.shields.io/github/stars/veyndor1/darica-eyetracking?style=social" alt="Stars" /></a>
</p>

<p align="center">
  <a href="#-kurulum">Kurulum</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="#-kullanim">Kullanim</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="#-mimari">Mimari</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="#-raspberry-pi">Raspberry Pi</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="#-katki">Katki</a>
</p>

<br/>

## Hakkinda

Darica Eye Tracking, siradan bir webcam ile goz hareketlerini takip edip mouse imlecini kontrol eden bir yazilim. Cift kirpmayla tiklama yapiyor, gozle taranabilen bir ekran klavyesiyle yazi yazmanizi sagliyor.

Ozel donanim gerekmiyor. Webcam ve Python yeterli.

Projenin ilk hedefi fiziksel engelli bireylerin bilgisayara erisimi, ama goz takibi uzerine deney yapmak isteyen herkes kullanabilir.

<br/>

<table>
<tr>
<td width="50%">

### Goz Takibi
- Iris pozisyonu + bas acisi ile mouse kontrolu
- 13 noktali kisisel kalibrasyon
- TPS ve Ridge regresyon ile bakis haritalama
- Kalman + One Euro filtre ile titresim bastirma
- Adaptif dead zone (yavas hareketlerde kararlililik)

</td>
<td width="50%">

### Kirpma ve Klavye
- Cift kirpma = sol tiklama
- Tarama klavyesi (goz kirpmasiyla harf secimi)
- Turkce kelime tahmini
- Adaptif EAR esigi (asagi bakista yalanci kirpma engeli)
- 2 sn goz kapatma ile klavye acma/kapama

</td>
</tr>
</table>

<br/>

## Mimari

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Sistem Mimarisi" width="100%" />
</p>

<details>
<summary>Teknik detaylar</summary>

<br/>

**Goz takibi:** MediaPipe Face Landmarker 478 yuz noktasi cikartiyor. Iris landmarklari (468-477) ve bas donusum matrisi, Ridge regresyon + TPS interpolasyonu ile kalibre edilmis ekran koordinatlarina donusuyor.

**Filtreleme:** Uc katman calisiyor sirayla. One Euro Filter dusuk gecikmeli yumusatma yapiyor, Kalman Filter durum tahmini ve gurultu azaltiyor, Adaptif Dead Zone ise yavas hareketlerde gereksiz titresimleri yok sayiyor.

**Kirpma algilama:** Eye Aspect Ratio (EAR) uzerinden calisiyor. Esik adaptif, surekli guncelleniyor. Asagi bakarken iris Y koordinatini kontrol ederek yalanci kirpmalari engelliyor.

</details>

<br/>

## Kurulum

> **Gereksinimler:** Python 3.9+, webcam, Windows 10/11 veya Raspberry Pi OS

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

`face_landmarker.task` repoda mevcut. Guncellemek isterseniz:

```bash
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

<br/>

## Kullanim

```bash
# Windows
python gazetracking.py

# Raspberry Pi
python rasbperrypi.py
```

Program acilinca 13 noktali kalibrasyon ekrani geliyor. Her noktaya sirayla bakin; sistem goz modelinizi olustursun. Kalibrasyon sirasinda hem basinizi hem gozlerinizi noktaya dogru cevirin.

### Kisayollar

| Tus | Ne yapar |
|:---:|----------|
| `q` | Cikis |
| `c` | Yeniden kalibrasyon |
| `p` | Mouse kontrolunu ac/kapat |
| `b` | Tiklamayi ac/kapat |

### Tarama klavyesi

Gozlerinizi 2 saniye kapali tutunca tarama klavyesi acilir (ayni sekilde kapatilir):

| Hareket | Sonuc |
|---------|-------|
| Tek kirpma | Sonraki harfe gec |
| Cift kirpma | Secili harfi yaz |
| ~1 sn goz kapali | Sonraki satira atla |
| ~2 sn goz kapali | Klavyeyi ac/kapat |

<br/>

## Raspberry Pi

`rasbperrypi.py` Raspberry Pi icin yazilmis surum. Farklari:

- **Kamera:** Picamera2 API veya V4L2 backend
- **Ekran boyutu:** `xrandr` ya da `tkinter` ile otomatik tespit
- **Giris kontrolu:** `pynput` kutuphanesi (X11 ortami gerekli)

Raspberry Pi 4 veya ustu onerilir, MediaPipe baya islemci istiyor.

<br/>

## Proje yapisi

```
darica-eyetracking/
├── gazetracking.py            # Ana uygulama (Windows)
├── rasbperrypi.py             # Raspberry Pi surumu
├── face_landmarker.task       # MediaPipe yuz modeli
├── requirements.txt           # Bagimliliklar (Windows)
├── requirements-rpi.txt       # Bagimliliklar (Raspberry Pi)
├── docs/assets/               # Logo, banner, diyagramlar
├── .github/
│   ├── workflows/ci.yml       # CI pipeline
│   ├── ISSUE_TEMPLATE/        # Issue sablonlari
│   └── pull_request_template.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── LICENSE
```

<br/>

## Katki

Hata buldunuz, ozellik fikriniz var ya da kod gondermek istiyorsaniz [CONTRIBUTING.md](CONTRIBUTING.md) dosyasina bakin. Her turlu katki kabul edilir.

<br/>

## Lisans

[MIT](LICENSE)

<br/>

---

<p align="center">
  <sub>Darica Eye Tracking, goz takibi ile erisilebilirlik uzerine acik kaynak bir projedir.</sub>
</p>
