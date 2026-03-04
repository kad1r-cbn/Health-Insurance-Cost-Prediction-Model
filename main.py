#default kütüphaneler
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib

# Sklearn Modülleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ayarlar
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")

output_dir = "Proje_Ciktilari"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_check_data(dataset):
    print("Loading data...")
    df = pd.read_csv(dataset)
    print(f"Data Size: {df.shape}")
    print(df.head())
    return df


def perform_eda(df):
    print("--- Görselleştirme Yapılıyor ve Kaydediliyor ---")
    # 1. Hedef Değişken Dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True, color='blue')
    plt.title('Sigorta Masraflari Dagilimi')
    plt.savefig(f"{output_dir}/1_charges_dagilimi.png")
    plt.close()

    # 2. Korelasyon Matrisi
    df_temp = df.copy()
    le = LabelEncoder()
    for col in df_temp.select_dtypes(include='object').columns:
        df_temp[col] = le.fit_transform(df_temp[col])

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_temp.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Ozellikler Arasi Korelasyon')
    plt.savefig(f"{output_dir}/2_korelasyon_haritasi.png")
    plt.close()

    # 3. Sigara ve Masraf İlişkisi
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
    plt.title('Sigara Icen vs Icmeyen Masraf')
    plt.savefig(f"{output_dir}/3_sigara_etkisi.png")
    plt.close()

    print(f"Grafikler '{output_dir}' klasörüne kaydedildi.")


def preprocess_data(dataframe):
    print("--- Veri Ön İşleme ---")
    df = dataframe.copy()
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    return df


def train_and_save_models(X_train, y_train, X_test, y_test):
    print("--- 4. Modeller Eğitiliyor ve En İyisi Seçiliyor ---")

    # Scale işlemi (Lineer modeller için)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    best_score = -np.inf
    best_model = None
    best_model_name = ""

    txt_output = "MODEL SONUCLARI RAPORU\n======================\n"

    for name, model in models.items():
        # Lineer regresyon scale ister, diğerleri istemez ama kontrol edelim
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append([name, rmse, r2, mae])
        txt_output += f"\nModel: {name}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}\nMAE: {mae:.4f}\n----------------------"

        # En iyi modeli hafızada tut
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    # Sonuçları Tablolaştır
    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2 Score", "MAE"])
    results_df = results_df.sort_values(by="R2 Score", ascending=False)

    print("\n" + txt_output)
    print(f"\n>>> BEST MODEL: {best_model_name} (R2: {best_score:.4f})")

    # EN İYİ MODELİ KAYDET (.pkl)
    pkl_filename = f"{output_dir}/best_insurance_model.pkl"
    joblib.dump(best_model, pkl_filename)
    print(f">>> Model dosyasi '{pkl_filename}' olarak kaydedildi.")

    # Raporu kaydet
    with open(f"{output_dir}/model_sonuclari.txt", "w") as f:
        f.write(txt_output)
        f.write(f"\n\nSECILEN EN IYI MODEL: {best_model_name}\n")
        f.write("\nKarsilastirma Tablosu:\n")
        f.write(results_df.to_string())

    return best_model


def demo_prediction(model, X_test, y_test):
    print("\n--- 5. Canli Tahmin Senaryosu (Demo) ---")
    print("Test veri setinden rastgele 3 kisi seciliyor ve model tahmin yapiyor...\n")

    sample_indices = X_test.sample(3, random_state=42).index

    for i in sample_indices:
        real_value = y_test.loc[i]
        # Tek satırlık tahmin için reshape gerekebilir
        row_data = X_test.loc[[i]]
        prediction = model.predict(row_data)[0]

        diff = prediction - real_value
        print(f"Musteri ID: {i}")
        print(f"Gercek Sigorta Masrafi : {real_value:.2f} $")
        print(f"Modelin Tahmini        : {prediction:.2f} $")
        print(f"Fark (Hata)            : {diff:.2f} $\n")


def main():
    data_path = 'data_set/insurance.csv'

    # 1. Yükleme
    df = load_check_data(data_path)

    # 2. EDA
    perform_eda(df)

    # 3. İşleme
    df_processed = preprocess_data(df)

    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train and save
    best_model = train_and_save_models(X_train, y_train, X_test, y_test)

    # 5. Sample

    if "Linear" not in str(type(best_model)):
        demo_prediction(best_model, X_test, y_test)

    print("--- PROJE BASARIYLA TAMAMLANDI ---")


if __name__ == "__main__":
    main()