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
    """Enhanced data loading dengan validasi"""
    try:
        # Cek jika file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
            
        print(f"Mengambil data dari path: {input_path}")
        df = pd.read_csv(input_path)
        
        # Validasi dataset
        if df.empty:
            raise ValueError("Dataset kosong!")
            
        print(f"Dataset berhasil dimuat: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        sys.exit(1)

def train_model(df):
    """Enhanced training dengan logging metrics lebih detail"""
    print("\n" + "="*50)
    print("MODEL TRAINING DENGAN MLFLOW AUTOLOG")
    print("="*50)
    
    # Feature dan target separation
    target_columns = [col for col in df.columns if col.startswith('Species_')]
    feature_columns = [col for col in df.columns if col not in target_columns and col != 'Unnamed: 0']
    
    if not target_columns:
        print("Kolom target 'Species_' tidak ditemukan.")
        return None

    X = df[feature_columns]
    y_ohe = df[target_columns] 
    y = y_ohe.idxmax(axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Setup MLflow
    experiment_name = "Iris_Classification_CI_Skilled"
    mlflow.set_experiment(experiment_name)
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("dataset", "Iris Processed")
        
        # Train model
        model = LogisticRegression(solver='lbfgs', multi_class='auto', 
                                 max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Additional logging
        print(f"\nAkurasi pada data uji: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log model signature (optional)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        return run.info.run_id

def main():
    parser = argparse.ArgumentParser(description="Training model Iris Classification")
    parser.add_argument("--input_data", type=str, required=True,
                       help="Path ke file CSV data yang sudah diproses")
    
    args = parser.parse_args()
    
    print("Memulai proses training...")
    df_processed = load_processed_data(args.input_data)
    
    if df_processed is not None:
        run_id = train_model(df_processed)
        if run_id:
            print(f"\nTraining selesai! Run ID: {run_id}")
        else:
            print("\nTraining gagal!")

if __name__ == "__main__":
    main()