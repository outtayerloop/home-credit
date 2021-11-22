import mlflow


def get_fit_mlflow_model(model):
    """Return latest trained model with maximum accuracy from MLFlow that was loaded in memory.

    Keyword arguments:
    model -- chosen model between gb (Gradient Boosting), xgb (XGBoost) or rf (Random Forest)
    """
    max_acc_run_id = get_max_acc_model_run_id(model)
    logged_model = 'runs:/' + max_acc_run_id + '/model'
    loaded_model = mlflow.sklearn.load_model(logged_model)
    return loaded_model


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
        return 'XGBClassifier'
    else:
        return 'RandomForestClassifier'