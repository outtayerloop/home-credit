# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="5ed90e31"
# # XGboost model train

# + id="a5013a5f"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import logging
import mlflow
from urllib.parse import urlparse
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# + [markdown] id="d1d6ca88"
# ## Splitting dataset into train and test

# + id="738f427f"
def get_split_train_data():
    """Return a tuple containing split train data into X_train X_test, y_train and y_test."""
    df = pd.read_csv('../../../../data/processed/processed_application_train.csv')
    train, test = train_test_split(df)
    X_train = train.drop(['TARGET'], axis=1)
    X_test = test.drop(['TARGET'], axis=1)
    y_train = train[['TARGET']]
    y_test = test[['TARGET']]
    return X_train, X_test, y_train, y_test


# + [markdown] id="931a8c69"
# ## Adding MLFLow workflow

# + [markdown] id="mXVizP3vXWdc"
# ### Configuring logs

# + id="ElavYC5wXOrF"
def get_configured_logger():
    """Return a logger for console outputs configured to print warnings."""
    logging.basicConfig(level=logging.WARN)
    return logging.getLogger(__name__)


# + [markdown] id="hFmZgG_hXvgi"
# ### Training model on split data

# + id="Jod5BBLEX0qJ"
def train_xgboost_classifier(X_train, y_train):
  """Return XGBClassifier fit on input ndarrays X_train and y_train.

  Keyword arguments:
  X_train -- ndarray containing all train columns except target column
  y_train -- ndarray target column values to train the model
  """
  clf = xgb.XGBClassifier(scale_pos_weight=(1 - y_train.values.mean()), n_jobs=-1)
  grid_search = GridSearchCV(clf,
                    {'max_depth': [5, 10], 'min_child_weight': [5, 10], 'n_estimators': [25]},
                    n_jobs=-1, cv=5, scoring='accuracy')
  grid_search.fit(X_train.values, y_train.values)
  clf.set_params(**grid_search.best_params_)
  clf = clf.fit(X_train, y_train)
  return clf


# + [markdown] id="ZtpOwQmScTqs"
# ### Getting model evaluation metrics

# + id="30ac1411"
def eval_metrics(actual, pred):
    """Return a tuple containing model classification accuracy, confusion matrix and f1_score.

  Keyword arguments:
  actual -- ndarray y_test containing true target values
  pred -- ndarray of the predicted target values by the model
  """
    accuracy = accuracy_score(actual, pred)
    conf_matrix = confusion_matrix(actual, pred)
    f_score = f1_score(actual, pred)
    return accuracy, conf_matrix, f_score


# + id="xlD2NDs7Yl52"
def get_model_evaluation_metrics(clf, X_test, y_test):
    """Return a tuple containing model classification accuracy, confusion matrix, f1_score  and ROC area under the curve score..
  
  Keyword arguments:
  clf -- classifier model
  X_test -- ndarray containing all test columns except target column
  y_test -- ndarray target column values to test the model
  """
    predicted_repayments = clf.predict(X_test)
    (accuracy, conf_matrix, f_score) = eval_metrics(y_test, predicted_repayments)
    xgb_probs = clf.predict_proba(X_test)
    xgb_probs = xgb_probs[:, 0]  # keeping only the first class (repayment OK)
    xgb_roc_auc_score = roc_auc_score(y_test, xgb_probs)
    return accuracy, conf_matrix, f_score, xgb_roc_auc_score


# + [markdown] id="xIhVvhSMcbRN"
# ### Tracking model on MLFLow

# + id="o82vjjLVb1NC"
def track_model_params(clf):
    """Log model params on MLFlow UI.

  Keyword arguments:
  clf -- classifier model
  """
    clf_params = clf.get_params()
    for param in clf_params:
        param_value = clf_params[param]
        mlflow.log_param(param, param_value)


# + id="bFudJAzUcjzI"
def track_model_metrics(clf, X_test, y_test):
    """Log model metrics on MLFlow UI.
  
  Keyword arguments:
  clf -- classifier model
  X_test -- ndarray containing all test columns except target column
  y_test -- ndarray target column values to test the model
  """
    (accuracy, conf_matrix, f_score, xgb_roc_auc_score) = get_model_evaluation_metrics(clf, X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f_score)
    mlflow.log_metric('roc_auc_score', xgb_roc_auc_score)
    tn, fp, fn, tp = conf_matrix.ravel()
    mlflow.log_metric('true_negatives', tn)
    mlflow.log_metric('false_positives', fp)
    mlflow.log_metric('false_negatives', fn)
    mlflow.log_metric('true_positives', tp)


# + id="DpBgmHX9dXcv"
def track_model_version(clf):
    """Version model on MLFlow UI.

  Keyword arguments:
  clf -- classifier model
  """
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(clf, 'model', registered_model_name='XGBClassifier')
    else:
        mlflow.sklearn.log_model(clf, 'model')


# + pycharm={"name": "#%%\n"} id="bNg-yhC7uSeS"
def set_mlflow_run_tags():
    """Set current MLFlow run tags."""
    tags = {'model_name': 'XGBClassifier'}
    mlflow.set_tags(tags)


# + id="e862b8bd"
def train_and_track_model_in_mlflow():
    """Train model and track it with MLFLow"""
    (X_train, X_test, y_train, y_test) = get_split_train_data()
    logger = get_configured_logger()
    clf = train_xgboost_classifier(X_train, y_train)
    with mlflow.start_run():
        track_model_params(clf)
        track_model_metrics(clf, X_test, y_test)
        track_model_version(clf)
        set_mlflow_run_tags()


# + pycharm={"name": "#%%\n"} colab={"base_uri": "https://localhost:8080/"} id="-e1kKCzVuSeT" outputId="66804a05-b021-43c8-db76-a249c2185edd"
if __name__ == '__main__':
    train_and_track_model_in_mlflow()
