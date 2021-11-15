import mlflow
import pandas as pd


def predict():
    """Predict all data from test dataset using Gradient Boosting model for classification."""
    max_acc_run_id = get_max_acc_model_run_id()
    logged_model = 'runs:/' + max_acc_run_id + '/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    test = pd.read_csv('./data/processed/processed_application_test.csv').head()
    y_pred = loaded_model.predict(test)
    print(y_pred)


def get_max_acc_model_run_id():
    """Get latest trained Gradient Boosting model with maximum accuracy."""
    found_models = mlflow.search_runs(
        filter_string='tags.model_name = \'GradientBoostingClassifier\'',
        order_by=['attribute.start_time ASC']
    )
    max_acc = found_models['metrics.accuracy'].max()
    models_with_max_acc = found_models[found_models['metrics.accuracy'] == max_acc]
    max_acc_run_id = models_with_max_acc['run_id'].tail(1).values[0]
    return max_acc_run_id