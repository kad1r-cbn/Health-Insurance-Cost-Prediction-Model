# Health Insurance Cost Prediction Model

## 📌 Proje Amacı
Bu proje, kişilerin demografik ve sağlık bilgilerine (yaş, cinsiyet, VKİ, çocuk sayısı, sigara kullanımı, bölge) dayanarak sağlık sigortası masraflarını tahmin etmeyi amaçlayan uçtan uca bir makine öğrenmesi çalışmasıdır. 

*Not: Bu proje, İBB Tech Istanbul tarafından düzenlenen 40 saatlik Makine Öğrenmesi Bootcamp programı kapsamında geliştirilmiştir.*

## 🛠️ Kullanılan Teknolojiler
* **Programlama Dili:** Python
* **Veri Manipülasyonu & Analizi:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn [Eğer XGBoost veya LightGBM kullandıysan buraya ekle]
* **Model Kayıt:** Pickle

## 📂 Proje Dosyaları
* `main.py`: Veri ön işleme, model eğitimi, hiperparametre optimizasyonu ve test süreçlerini içeren ana betik.
* `best_insurance_model.pkl`: Eğitilmiş ve optimize edilmiş en iyi performansı gösteren makine öğrenmesi modeli.
* `model_sonuclari.txt`: Modelin test verisi üzerindeki metriklerini içeren çıktı dosyası.

## ⚙️ Geliştirme Süreci
1. **Veri Ön İşleme:** Kategorik değişkenlerin (cinsiyet, sigara içme durumu, bölge) encode edilmesi ve sayısal değişkenlerin ölçeklendirilmesi.
2. **Model Seçimi ve Eğitimi:** Birden fazla regresyon algoritmasının denenmesi ve çapraz doğrulama (cross-validation) ile sonuçların karşılaştırılması.
3. **Hiperparametre Optimizasyonu:** En iyi model üzerinde Grid Search / Random Search uygulanarak parametrelerin ayarlanması.
4. **Modelin Kaydedilmesi:** Nihai modelin `pickle` kullanılarak dışa aktarılması.

## 📊 Model Sonuçları
Modelin test veri seti üzerindeki performansı aşağıdaki gibidir (Detaylar `model_sonuclari.txt` dosyasında bulunabilir):
* **R2 Score:** [Buraya txt dosyasındaki R2 değerini yaz]
* **Root Mean Squared Error (RMSE):** [Buraya txt dosyasındaki RMSE değerini yaz]
* **Mean Absolute Error (MAE):** [Buraya txt dosyasındaki MAE değerini yaz]

## 🚀 Kurulum ve Kullanım
Projeyi lokalinizde çalıştırmak için:

1. Repoyu klonlayın:
```bash
git clone [https://github.com/](https://github.com/)[KULLANICI_ADIN]/insurance-cost-prediction.git
cd insurance-cost-prediction
