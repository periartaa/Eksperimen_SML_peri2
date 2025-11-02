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
        print(f"Shape dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {input_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
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
        print("❌ Kolom target 'Species_' tidak ditemukan.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    X = df[feature_columns]
    y_ohe = df[target_columns] 
    
    # Mengkonversi OHE ke Label Tunggal (1D Array)
    y = y_ohe.idxmax(axis=1) 
    
    # Pisahkan data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    
    # AKTIVASI MLFLOW AUTOLOG
    mlflow.sklearn.autolog()
    
    # Menentukan nama eksperimen
    experiment_name = "Iris_Classification_CI"
    mlflow.set_experiment(experiment_name)
    
    # Memulai run MLflow
    with mlflow.start_run() as run:
        # Menambahkan tag
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("data_split", "70-30")
        mlflow.set_tag("environment", "CI/CD Pipeline")
        
        # Inisialisasi dan latih model
        model = LogisticRegression(
            solver='lbfgs', 
            multi_class='auto', 
            max_iter=1000, 
            random_state=42
        )
        
        model.fit(X_train, y_train) 
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrik tambahan
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        print("\n✅ Model berhasil dilatih.")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("\n📊 Evaluasi Model:")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1 Score:  {f1:.4f}")
        
    print("\n🎯 Model Training Selesai!")

def main():
    """
    Fungsi utama untuk menjalankan pipeline modeling.
    """
    # Ambil path dari argument atau gunakan default
    if len(sys.argv) > 1:
        processed_file_path = sys.argv[1]
    else:
        # 🌟 PERBAIKAN: Cek beberapa lokasi yang mungkin 🌟
        possible_paths = [
            'namadataset_preprocessing/iris_processed.csv',
            'iris_processed.csv',
            '../iris_processed.csv'
        ]
        
        processed_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                processed_file_path = path
                break
        
        if processed_file_path is None:
            print("❌ File iris_processed.csv tidak ditemukan di lokasi manapun!")
            print(f"Lokasi yang dicek: {possible_paths}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files available: {os.listdir('.')}")
            sys.exit(1)
    
    print(f"📂 Loading data from: {processed_file_path}")
    
    # 1. Load data
    df_processed = load_processed_data(processed_file_path)
    
    # 2. Train model dengan MLflow Autolog
    train_model(df_processed)

if __name__ == "__main__":
    main()