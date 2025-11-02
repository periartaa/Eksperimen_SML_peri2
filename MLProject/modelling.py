# MLProject/modelling.py

import pandas as pd
import sys
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

def load_processed_data(input_path):
    """
    Memuat dataset yang sudah diproses dari file CSV.
    """
    try:
        print(f"Mengambil data dari path: {input_path}")
        df = pd.read_csv(input_path)
        print("✅ Dataset processed berhasil dimuat")
        return df
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_path}.")
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
    feature_columns = [col for col in df.columns if col not in target_columns and col != 'Unnamed: 0']
    
    if not target_columns:
        print("❌ Kolom target 'Species_' tidak ditemukan.")
        return

    X = df[feature_columns]
    y_ohe = df[target_columns] 
    y = y_ohe.idxmax(axis=1) # Konversi OHE ke Label Tunggal
    
    # Pisahkan data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # AKTIVASI MLFLOW AUTOLOG
    mlflow.sklearn.autolog()
    
    experiment_name = "Iris_Classification_CI_Skilled"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "Logistic Regression")
        
        # Inisialisasi dan latih model
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        model.fit(X_train, y_train) 
        
        y_pred = model.predict(X_test)
        
        print("\n✅ Model berhasil dilatih dan dicatat.")
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAkurasi pada data uji: {accuracy:.4f}")
        
    print("\n🎯 Model Training Selesai!")


def main():
    # Inisialisasi Argument Parser
    parser = argparse.ArgumentParser(description="Menjalankan training model Iris Classification.")
    # Argumen yang diterima dari MLProject
    parser.add_argument("--input_data", type=str, required=True, 
                        help="Path relatif ke file CSV data yang sudah diproses.")
    
    args, unknown = parser.parse_known_args() # Menggunakan parse_known_args untuk mengatasi argumen tambahan dari MLflow
    
    df_processed = load_processed_data(args.input_data)
    
    if df_processed is not None:
        train_model(df_processed)

if __name__ == "__main__":
    main()