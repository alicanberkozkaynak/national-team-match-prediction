# Milli Maç Tahmin Web Arayüzü

## Çalıştırma

PowerShell veya CMD içinde proje klasöründe:

```powershell
python app.py
```

Ardından tarayıcıda şunu aç:

```text
http://127.0.0.1:8000
```

Windows'ta alternatif olarak `start_web.bat` dosyasına çift tıklayabilirsin.

## Gerekli paketler

Web sunucusu yalnızca Python standart kütüphanesini kullanır. Modelin kendisi için mevcut ortamındaki `joblib`, `numpy`, `pandas`, `scipy`, `scikit-learn` ve `xgboost` paketleri gereklidir.

## Dosyalar

- `app.py`: Yerel HTTP sunucusu ve tahmin API'si
- `static/index.html`: Arayüz
- `static/styles.css`: Tasarım
- `static/app.js`: Form, API isteği ve sonuçların gösterimi
- `national_team_score_model.py`: Mevcut model/tahmin mantığı
- `national_team_model.joblib`: Eğitilmiş model
