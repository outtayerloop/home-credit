import subprocess
from src.features import build_features as bf

if __name__ == '__main__':
    bf.build_features()
    subprocess.run('mlflow run ./src/models/train/gradient_boosting')