import mlflow
import pandas as pd


if __name__ == '__main__':
    truc = mlflow.search_runs(
        filter_string = 'tags.model_name = \'GradientBoostingClassifier\'',
        order_by=['attribute.start_time ASC']
    )
    gradient_boosting_train_latest_run_id = truc['run_id'].tail(1).values[0]
    logged_gradient_boosting_model = 'runs:/' + gradient_boosting_train_latest_run_id + '/model'
    loaded_radient_boosting_model = mlflow.pyfunc.load_model(logged_gradient_boosting_model)
    test = pd.read_csv('./data/processed/processed_application_test.csv').head()
    y_pred = loaded_radient_boosting_model.predict(test)
    print(y_pred)