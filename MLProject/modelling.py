import pandas as pd
import os
import sys
import argparse # Diperlukan untuk menerima parameter dari MLProject

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
        # Menampilkan path yang sedang dimuat
        print(f"Mengambil data dari path: {input_path}")
        df = pd.read_csv(input_path)
        print("✅ Dataset processed berhasil dimuat")
        return df
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_path}. Pastikan file data tersedia di path ini.")
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
    feature_columns = [col for col in df.columns if col not in target_columns and col != 'Unnamed: 0'] # Menghindari kolom indeks yang tidak terpakai
    
    if not target_columns:
        print("❌ Kolom target 'Species_' tidak ditemukan. Pastikan preprocessing sudah benar (One-Hot Encoding).")
        return

    X = df[feature_columns]
    y_ohe = df[target_columns] 
    
    # Mengkonversi OHE ke Label Tunggal (1D Array) untuk scikit-learn
    y = y_ohe.idxmax(axis=1) 
    
    # Pisahkan data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train (1D): {y_train.shape}") 
    print(f"Shape X_test: {X_test.shape}")
    
    # AKTIVASI MLFLOW AUTOLOG
    mlflow.sklearn.autolog()
    
    # Menentukan nama eksperimen
    experiment_name = "Iris_Classification_Basic"
    mlflow.set_experiment(experiment_name)
    
    # Memulai run MLflow
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("data_split", "70-30")
        
        # Inisialisasi dan latih model
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        model.fit(X_train, y_train) 
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        
        print("\n✅ Model berhasil dilatih.")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Data dan model dicatat otomatis oleh MLflow autolog.")
        
        # Evaluasi manual (untuk output konsol)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAkurasi pada data uji: {accuracy:.4f}")
        
    print("\n🎯 Model Training Selesai! Cek MLflow Tracking UI (folder mlruns).")


def main(input_data_path):
    """
    Fungsi utama untuk menjalankan pipeline modeling, menerima path data.
    """
    
    # 1. Load data
    df_processed = load_processed_data(input_data_path)
    
    # 2. Train model dengan MLflow Autolog
    if df_processed is not None:
        train_model(df_processed)

if __name__ == "__main__":
    # Inisialisasi Argument Parser
    parser = argparse.ArgumentParser(description="Menjalankan training model Iris Classification.")
    # Menentukan argumen yang akan diterima dari MLProject
    parser.add_argument("--input_data", type=str, required=True, 
                        help="Path relatif ke file CSV data yang sudah diproses (e.g., namadataset_preprocessing/iris_processed.csv).")
    
    args = parser.parse_args()
    
    # Jalankan fungsi utama dengan path data yang diterima
    main(args.input_data)
