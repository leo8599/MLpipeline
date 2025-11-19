import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class WidsDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.target_col = 'DiagPeriodL90D'
        self.target_col_clean = 'DiagPeriod90'
        self.scaler = MinMaxScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        
    def load_data(self, filename):
        """Carga los datos desde un archivo CSV."""
        full_path = f"{self.data_path}/{filename}"
        df = pd.read_csv(full_path)
        print(f"Datos cargados: {df.shape}")
        return df

    def _create_bmi_categories(self, df):
        """Crea categorías de BMI basadas en la lógica del notebook."""
        # Imputar nulos temporalmente para categorización
        df['bmi_temp'] = df['bmi'].fillna(0)
        bins = [0, 18.5, 25, 30, df['bmi_temp'].max() + 1]
        labels = ['underweight', 'healthy', 'overweight', 'obese']
        
        df['bmi_category'] = pd.cut(df['bmi_temp'], bins=bins, labels=labels, right=False).astype(object)
        
        # Marcar los que eran nulos originalmente como 'not_specified'
        df.loc[df['bmi'].isnull(), 'bmi_category'] = 'bmi_not_specified'
        df.drop(columns=['bmi', 'bmi_temp'], inplace=True)
        return df

    def _create_age_categories(self, df):
        """Crea categorías de Edad."""
        bins = [0, 30, 45, 60, 75, 120]
        labels = ['junior', 'young', 'adult', 'senior', 'old']
        
        df['age_category'] = pd.cut(df['patient_age'], bins=bins, labels=labels, right=False).astype(object)
        df.drop(columns=['patient_age'], inplace=True)
        return df

    def clean_and_transform(self, df, is_train=True):
        """
        Aplica limpieza, imputación y feature engineering.
        """
        # 1. Renombrar target si existe
        if self.target_col in df.columns:
            df = df.rename(columns={self.target_col: self.target_col_clean})

        # 2. Eliminar columnas con muchos nulos o irrelevantes
        cols_to_drop = ['metastatic_first_novel_treatment', 
                        'metastatic_first_novel_treatment_type', 
                        'patient_id']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        # 3. Imputación de Categóricas
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('not_specified')

        # 4. Imputación de Numéricas (con la media)
        num_cols = df.select_dtypes(include=[np.number]).columns
        # Excluir el target de la imputación si es set de entrenamiento
        cols_to_impute = [c for c in num_cols if c != self.target_col_clean]
        for col in cols_to_impute:
            df[col] = df[col].fillna(df[col].mean())

        # 5. Ingeniería de Características (BMI y Edad)
        if 'bmi' in df.columns:
            df = self._create_bmi_categories(df)
        if 'patient_age' in df.columns:
            df = self._create_age_categories(df)
            
        # Eliminar filas con nulos residuales en target (solo train)
        if is_train and self.target_col_clean in df.columns:
            df.dropna(subset=[self.target_col_clean], inplace=True)
            
        return df

    def prepare_for_training(self, df):
        """
        Realiza One-Hot Encoding y Escalado.
        Devuelve X_train, X_val, y_train, y_val
        """
        # Separar features y target
        y = df[self.target_col_clean]
        X = df.drop(columns=[self.target_col_clean])

        # One-Hot Encoding
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Escalado (MinMax como en el notebook)
        # Identificar columnas numéricas en el df ya codificado
        # Nota: Las dummies son uint8/bool, escalamos solo las que eran float/int originales
        # Para simplificar, escalaremos todas las columnas numéricas resultantes
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_encoded), 
                                columns=X_encoded.columns, 
                                index=X_encoded.index)

        # División Train/Test (80/20)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)