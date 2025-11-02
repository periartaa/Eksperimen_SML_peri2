# MLProject/modelling.py

import pandas as pd
import sys
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn

def load_processed_data(input_path):
    """
    Memuat dataset yang sudah diproses dari file CSV.
    """
    try:
        print(f"Mengambil data dari path: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
            
        df = pd.read_csv(input_path)
        print(f" Dataset berhasil dimuat. Shape: {df.shape}")
        return df
        
    except FileNotFoundError as e:
        print(f" ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f" Error saat memuat dataset: {e}")
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
        print(" ERROR: Kolom target 'Species_' tidak ditemukan.")
        return

    X = df[feature_columns]
    y_ohe = df[target_columns] 
    y = y_ohe.idxmax(axis=1)
    
    print(f" Fitur: {X.shape[1]} kolom")
    print(f" Target classes: {y.unique()}")
    
    # Pisahkan data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Setup MLflow
    experiment_name = "Iris_Classification_CI_Skilled"
    mlflow.set_experiment(experiment_name)
    
    # AKTIVASI MLFLOW AUTOLOG
    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("workflow", "github_actions_ci")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)
        
        # Inisialisasi dan latih model
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
        model.fit(X_train, y_train) 
        
        y_pred = model.predict(X_test)
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"\n Model berhasil dilatih. Run ID: {run.info.run_id}")
        print(f" Akurasi pada data uji: {accuracy:.4f}")
        
    print("\n Model Training Selesai!")

def main():
    parser = argparse.ArgumentParser(description="Menjalankan training model Iris Classification.")
    parser.add_argument("--input_data", type=str, required=True, 
                        help="Path relatif ke file CSV data yang sudah diproses.")
    
    args, unknown = parser.parse_known_args()
    
    print("="*50)
    print("IRIS CLASSIFICATION TRAINING")
    print("="*50)
    print(f"Input data: {args.input_data}")
    
    df_processed = load_processed_data(args.input_data)
    
    if df_processed is not None:
        train_model(df_processed)

if __name__ == "__main__":
    main()