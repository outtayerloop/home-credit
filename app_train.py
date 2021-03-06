import subprocess
from src.features import build_features as bf


def train_models():
    """Pre-process both train and test CSVs in data/external, save them in data/processed,
     train models and track them with MLFlow."""
    bf.build_features()
    subprocess.run('mlflow run ./src/models/train/gradient_boosting.rst')
    subprocess.run('mlflow run ./src/models/train/xgboost')
    subprocess.run('mlflow run ./src/models/train/random_forest')


if __name__ == '__main__':
    train_models()