import mlflow
import pandas as pd


def predict(model, file_path):
    """Predict all data from test dataset using given  model for classification.

    Keyword arguments:
    model -- chosen model
    file_path -- CSV file path
    """
    max_acc_run_id = get_max_acc_model_run_id(model)
    logged_model = 'runs:/' + max_acc_run_id + '/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    try:
        test = pd.read_csv(file_path).head()  # ./data/processed/processed_application_test.csv
        y_pred = loaded_model.predict(test)
        print(y_pred)
    except Exception:
        print('Given CSV file path ' + file_path + ' not found.')


def get_max_acc_model_run_id(model):
    """Get latest trained model with maximum accuracy.

    Keyword arguments:
    model -- chosen model
    """
    model_tag = get_model_tag(model)
    found_models = mlflow.search_runs(
        filter_string='tags.model_name = \'' + model_tag + '\'',
        order_by=['attribute.start_time ASC']
    )
    max_acc = found_models['metrics.accuracy'].max()
    models_with_max_acc = found_models[found_models['metrics.accuracy'] == max_acc]
    max_acc_run_id = models_with_max_acc['run_id'].tail(1).values[0]
    return max_acc_run_id


def get_model_tag(model):
    """Get model MLFlow tag.

    Keyword arguments:
    model -- chosen model
    """
    if model == 'gb':
        return 'GradientBoostingClassifier'
    elif model == 'xgb':
        return 'XGBoost'
    else:
        return 'RandomForest'