import pandas as pd
from module_data import WidsDataProcessor
from module_ml import ModelTrainer

# Importar modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def main():
    print("=== Iniciando Pipeline WIDS 2024 ===")
    
    # 1. Carga y Procesamiento de Datos
    processor = WidsDataProcessor(data_path="data")
    
    try:
        df_raw = processor.load_data("training.csv")
    except FileNotFoundError:
        print("Error: No se encuentra 'data/training.csv'. Asegúrate de descargar los datos.")
        return

    print("Limpiando y transformando datos...")
    df_clean = processor.clean_and_transform(df_raw, is_train=True)
    
    print("Codificando y dividiendo datos (Train/Val)...")
    X_train, X_val, y_train, y_val = processor.prepare_for_training(df_clean)
    
    print(f"Dimensiones Train: {X_train.shape}, Val: {X_val.shape}")

    # 2. Configuración de Modelos
    # Diccionario de modelos a experimentar
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP (Neural Net)": MLPClassifier(max_iter=500, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    # 3. Entrenamiento y Tracking con MLflow
    trainer = ModelTrainer(experiment_name="WiDS_Model_Comparison")
    results = []

    for name, model in models.items():
        metrics = trainer.train_evaluate(model, name, X_train, X_val, y_train, y_val)
        results.append(metrics)

    # 4. Resumen Final
    print("\n=== Resumen de Resultados ===")
    results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    print(results_df)
    
    # Guardar resultados simples
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nResultados guardados en 'model_comparison_results.csv'")
    print("Ejecuta 'mlflow ui' en la terminal para ver los detalles del experimento.")

if __name__ == "__main__":
    main()