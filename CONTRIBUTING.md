# Katkı

Bu projeye katkıda bulunmak isterseniz aşağıdaki adımları takip edebilirsiniz.

## Hata bildirimi

Bir hata bulduysanız [yeni bir issue](https://github.com/veyndor1/darica-eyetracking/issues/new) açın. Şunları yazarsanız işi kolaylaştırır:

- İşletim sisteminiz ve Python sürümünüz
- Hatayı oluşturan adımlar
- Ne bekliyordunuz, ne oldu
- Kamera modeliniz (gerekliyse)

## Özellik önerisi

Bir fikriniz varsa yine issue açabilirsiniz. Ne için gerektiğini ve hangi durumda işe yarayacağını açıklarsanız değerlendirmesi daha kolay olur.

## Kod göndermek

1. Repoyu forklayın
2. Branch açın: `git checkout -b ozellik/aciklayici-isim`
3. Değişikliklerinizi yapın ve commitleyin
4. Push edin: `git push origin ozellik/aciklayici-isim`
5. Pull Request açın

## Geliştirme ortamı

```bash
git clone https://github.com/<kullanıcı-adınız>/darica-eyetracking.git
cd darica-eyetracking
pip install -r requirements.txt
```

## Commit mesajları

Açıklayıcı commit mesajları yazın:

```
fix: çift kırpma algılamada yalancı pozitif düzeltmesi
feat: kalibrasyon noktası sayısını yapılandırma seçeneği
docs: kurulum adımlarını güncelleme
```
