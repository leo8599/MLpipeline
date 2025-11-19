import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import time

class ModelTrainer:
    def __init__(self, experiment_name="WiDS_Datathon_2024"):
        # Configurar experimento de MLflow
        mlflow.set_experiment(experiment_name)

    def train_evaluate(self, model, model_name, X_train, X_val, y_train, y_val):
        """
        Entrena el modelo, calcula métricas y registra todo en MLflow.
        """
        with mlflow.start_run(run_name=model_name):
            print(f"--- Entrenando {model_name} ---")
            start_time = time.time()
            
            # Entrenamiento
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predicciones
            y_pred = model.predict(X_val)
            
            # Intentar obtener probabilidades para ROC-AUC
            try:
                y_prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob)
            except AttributeError:
                y_prob = None
                auc = 0.0  # Modelos que no soportan predict_proba

            # Métricas
            acc = accuracy_score(y_val, y_pred)
            
            print(f"Tiempo: {training_time:.4f}s | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

            # Logging en MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", auc)
            
            # Registrar el modelo (opcional, puede ocupar espacio)
            # mlflow.sklearn.log_model(model, model_name)
            
            return {
                "Model": model_name,
                "Accuracy": acc,
                "AUC": auc,
                "Time": training_time
            }