import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Avalia múltiplos modelos de regressão e retorna uma tabela com as métricas de desempenho.
    
    :param models: Dicionário de modelos para avaliar { 'nome_modelo': modelo }
    :param X_train: Conjunto de treino (features)
    :param y_train: Conjunto de treino (target)
    :param X_test: Conjunto de teste (features)
    :param y_test: Conjunto de teste (target)
    :return: DataFrame com as métricas de avaliação (MSE, RMSE, MAE, R²)
    """
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Modelo': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
    results_df = pd.DataFrame(results)
    
    return results_df
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Support Vector Machine (RBF Kernel)': SVR(kernel='rbf'),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'XGBoost': XGBRegressor(n_estimators=100),
    'SVM Linear': SVR(kernel='linear')
}

results_df = evaluate_models(models, X_training_regressor, y_training_regressor, X_test_regressor, y_test_regressor)
print(results_df)