import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

def load_processed_data(input_path):
    """
    Memuat dataset yang sudah diproses dari file CSV.
    """
    try:
        df = pd.read_csv(input_path)
        print("✅ Dataset processed berhasil dimuat")
        return df
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_path}. Pastikan langkah preprocessing sudah dijalankan.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saat memuat dataset: {e}")
        sys.exit(1)

def train_model(df):
    """
    Melatih model Klasifikasi (Logistic Regression) menggunakan MLflow autolog.
    """
    print("\n" + "="*50)
    print("MODEL TRAINING DENGAN MLFLOW AUTOLOG")
    print("="*50)
    
    # Memisahkan fitur (X) dan target (y)
    target_columns = [col for col in df.columns if col.startswith('Species_')]
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    if not target_columns:
        print("❌ Kolom target 'Species_' tidak ditemukan. Pastikan preprocessing sudah benar (One-Hot Encoding).")
        return

    X = df[feature_columns]
    # Target dalam format OHE (Multi-column)
    y_ohe = df[target_columns] 
    
    # 🌟 PERBAIKAN PENTING: Mengkonversi OHE ke Label Tunggal (1D Array) 🌟
    # .idxmax(axis=1) mengambil nama kolom dengan nilai 1, menghasilkan Series 1D
    y = y_ohe.idxmax(axis=1) 
    
    # Pisahkan data latih dan data uji
    # Gunakan 'y' (label tunggal) untuk stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train (1D): {y_train.shape}") # Shape sekarang adalah (105,)
    print(f"Shape X_test: {X_test.shape}")
    
    # AKTIVASI MLFLOW AUTOLOG
    mlflow.sklearn.autolog()
    
    # Menentukan nama eksperimen
    experiment_name = "Iris_Classification_Basic"
    mlflow.set_experiment(experiment_name)
    
    # Memulai run MLflow
    with mlflow.start_run() as run:
        # Menambahkan tag/informasi tambahan secara manual (opsional)
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("data_split", "70-30")
        
        # Inisialisasi dan latih model
        # multi_class='auto' bisa menangani 1D target.
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        
        # model.fit sekarang menerima y_train (1D array), sehingga error teratasi
        model.fit(X_train, y_train) 
        
        # Prediksi dan evaluasi (autolog akan mencatat metrik dasar)
        y_pred = model.predict(X_test)
        
        print("\n✅ Model berhasil dilatih.")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Data dan model dicatat otomatis oleh MLflow autolog.")
        
        # Evaluasi manual (untuk output konsol)
        # y_test dan y_pred di sini sudah dalam format label tunggal (1D)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAkurasi pada data uji: {accuracy:.4f}")
        
    print("\n🎯 Model Training Selesai! Cek MLflow Tracking UI.")

# Di modelling.py (Menggunakan Path Absolut untuk menghindari masalah)
def main():
    """
    Fungsi utama untuk menjalankan pipeline modeling.
    """
    
    # 🌟 GANTI DENGAN PATH LOKASI SEBENARNYA 🌟
    # Sesuaikan path ini jika Anda ingin mengarahkan ke lokasi spesifik.
    processed_file_path = r'C:\Peri\DICODING\ACCOUNT StudentPro\Eksperimen_SML_periart\preprocessing\iris_preprocessing\iris_processed.csv' # Sesuaikan ini!
    
    # 1. Load data
    df_processed = load_processed_data(processed_file_path)
    
    # 2. Train model dengan MLflow Autolog
    train_model(df_processed)

if __name__ == "__main__":
    main()