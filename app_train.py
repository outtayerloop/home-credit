import subprocess
from src.features import build_features as bf
from src.visualization import xgboost_xai as xai
from src.models import retrieve_fit_model as rfm
import pandas as pd


def train_models():
    """Pre-process both train and test CSVs in data/external, save them in data/processed,
     train models and track them with MLFlow."""
    #bf.build_features()
    #subprocess.run('mlflow run ./src/models/train/gradient_boosting')
    #subprocess.run('mlflow run ./src/models/train/xgboost')
    #subprocess.run('mlflow run ./src/models/train/random_forest')
    shap_values, explainer = get_xgboost_model_shap_values_and_explainer()
    test = pd.read_csv('./data/processed/processed_application_test.csv')
    xai.vizualize_single_line_xai(explainer, test.iloc[10], 10, shap_values)


def get_xgboost_model_shap_values_and_explainer():
    """Display fit XGBoost SHAP values."""
    fit_xgb = rfm.get_fit_mlflow_model('xgb')
    shap_values, explainer = xai.get_fit_model_shapley_values_and_explainer(fit_xgb)
    return shap_values, explainer


if __name__ == '__main__':
    train_models()