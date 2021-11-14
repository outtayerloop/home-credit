import os
from src.features import build_features as bf

if __name__ == '__main__':
    bf.build_features()
    os.system('mlflow run ./src/models/train/gradient_boosting')