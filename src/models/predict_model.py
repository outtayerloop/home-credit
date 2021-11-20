import pandas as pd
from src.models import retrieve_fit_model as rfm


def predict(model, file_path):
    """Predict all data from test dataset using given  model for classification.

    Keyword arguments:
    model -- chosen model
    file_path -- CSV file path
    """
    loaded_model = rfm.get_fit_mlflow_model(model)
    try:
        test = pd.read_csv(file_path).head()  # ./data/processed/processed_application_test.csv
        y_pred = loaded_model.predict(test)
        print(y_pred)
    except Exception:
        print('Given CSV file path ' + file_path + ' not found.')