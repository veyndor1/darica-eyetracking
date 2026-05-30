# Katki rehberi

## Hata bildirimi

Bir hata bulduyseniz [yeni bir issue](https://github.com/veyndor1/darica-eyetracking/issues/new?template=bug_report.yml) acin. Su bilgileri eklemeniz isi kolaylastirir:

- Isletim sistemi ve Python surumu
- Hatayi olusturan adimlar
- Ne bekliyordunuz, ne oldu
- Kamera modeli (gerekirse)

## Ozellik onerisi

Bir fikriniz varsa [feature request](https://github.com/veyndor1/darica-eyetracking/issues/new?template=feature_request.yml) acin. Ne icin gerektigini ve hangi durumda ise yarayacagini yazin.

## Kod gondermek

1. Repoyu forklayin
2. Branch acin: `git checkout -b feature/ozellik-adi`
3. Degisikliklerinizi yapin, commitleyin
4. Push edin: `git push origin feature/ozellik-adi`
5. Pull Request acin

## Gelistirme ortami

```bash
git clone https://github.com/<kullanici-adiniz>/darica-eyetracking.git
cd darica-eyetracking
pip install -r requirements.txt
```

## Kod stili

- Degisken ve fonksiyon adlari aciklayici olsun
- Karmasik yerlere kisa yorum ekleyin
- Mevcut stile uyun

## Commit mesajlari

```
fix: cift kirpma algilamada yalanci pozitif duzeltmesi
feat: kalibrasyon noktasi sayisini yapilandirma secenegi
docs: kurulum adimlarini guncelleme
```

## Soru sormak

[Issue](https://github.com/veyndor1/darica-eyetracking/issues) acabilirsiniz.
