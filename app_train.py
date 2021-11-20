import subprocess
from src.features import build_features as bf
from src.visualization import xgboost_xai as xai
from src.models import retrieve_fit_model as rfm


def train_models():
    """Pre-process both train and test CSVs in data/external, save them in data/processed,
     train models and track them with MLFlow."""
    bf.build_features()
    subprocess.run('mlflow run ./src/models/train/gradient_boosting')
    subprocess.run('mlflow run ./src/models/train/xgboost')
    display_xgboost_model_shap_values()


def display_xgboost_model_shap_values():
    """Display fit XGBoost SHAP values."""
    fit_xgb = rfm.get_fit_mlflow_model('xgb')
    shap_values = xai.get_fitted_model_shapley_values(fit_xgb)
    print(shap_values)


if __name__ == '__main__':
    train_models()