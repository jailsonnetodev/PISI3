import shap
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Exemplo: usando dados de treinamento
# X_train = suas_features_de_treinamento
# y_train = seu_target_de_treinamento

# Treinando o Random Forest
rf = RandomForestClassifier()
rf.fit(X_training, y_training)

# Treinando a Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_training, y_training)

# Treinando o KNN
knn = KNeighborsClassifier()
knn.fit(X_training, y_training)

# Calculando valores SHAP para Random Forest
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_training)

# Calculando valores SHAP para Decision Tree
explainer_dt = shap.TreeExplainer(dt)
shap_values_dt = explainer_dt.shap_values(X_training)

# Para KNN, usamos KernelExplainer
explainer_knn = shap.KernelExplainer(knn.predict_proba, X_training)
shap_values_knn = explainer_knn.shap_values(X_training)

# Função para calcular a importância média absoluta das características
def calculate_mean_shap_importance(shap_values, feature_names):
    mean_shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap_importance
    }).sort_values(by='importance', ascending=False)
    return importance_df

# Calculando a importância média das características
rf_importance_df = calculate_mean_shap_importance(shap_values_rf[1], X_training.columns)
dt_importance_df = calculate_mean_shap_importance(shap_values_dt[1], X_training.columns)
knn_importance_df = calculate_mean_shap_importance(shap_values_knn[1], X_training.columns)

# Função para plotar a importância usando Plotly
def plot_feature_importance_plotly(importance_df, model_name):
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title=f'Feature Importance - {model_name}', height=600)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.show()

# Plotando a importância das características para cada modelo
plot_feature_importance_plotly(rf_importance_df, 'Random Forest')
plot_feature_importance_plotly(dt_importance_df, 'Decision Tree')
plot_feature_importance_plotly(knn_importance_df, 'KNN')