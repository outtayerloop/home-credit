import pandas as pd
import json


def get_prediction_request_json_body():
    test = pd.read_csv('./data/processed/preprocessed_application_test.csv')
    columns = list(test.columns)
    data = list(test.iloc[0, :].values)
    body = {
        'columns': columns,
        'data': [data]
    }
    print(json.dumps(body))


if __name__ == '__main__':
    get_prediction_request_json_body()