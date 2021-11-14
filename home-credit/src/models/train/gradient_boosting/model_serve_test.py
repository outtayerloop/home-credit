import pandas as pd
import json


def get_prediction_request_json_body():
    test = pd.read_csv('./data/processed/preprocessed_application_test.csv')
    columns = list(test.columns)
    data = test.to_numpy().tolist()
    body = {
        'columns': columns,
        'data': data
    }
    with open('./src/models/train/gradient_boosting/prediction_request_body.json', 'w', encoding='utf-8') as file:
        json.dump(body, file, indent=4)


if __name__ == '__main__':
    get_prediction_request_json_body()