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
#     language: python
#     name: python3
# ---

# # Random Forest model train

# + pycharm={"name": "#%%\n"}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import logging
import mlflow
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier


# -

# ## Splitting dataset into train and test

# + pycharm={"name": "#%%\n"}
def get_split_train_data():
  """Return a tuple containing split train data into X_train X_test, y_train and y_test."""
  df = pd.read_csv('../../../../data/processed/processed_application_train.csv')
  train, test = train_test_split(df)
  X_train = train.drop(['TARGET'], axis=1)
  X_test = test.drop(['TARGET'], axis=1)
  y_train = train[['TARGET']]
  y_test = test[['TARGET']]
  return X_train, X_test, y_train, y_test


# -

# ## Adding MLFLow workflow

# ### Configuring logs

# + pycharm={"name": "#%%\n"}
def get_configured_logger():
  """Return a logger for console outputs configured to print warnings."""
  logging.basicConfig(level=logging.WARN)
  return logging.getLogger(__name__)


# -

# ### Training model on split data

# + pycharm={"name": "#%%\n"}
def train_random_forest_classifier(X_train, y_train):
  """Return RandomForestClassifier fit on input ndarrays X_train and y_train.

  Keyword arguments:
  X_train -- ndarray containing all train columns except target column
  y_train -- ndarray target column values to train the model
  """
  clf = RandomForestClassifier(criterion='gini', n_estimators=300)
  clf = clf.fit(X_train, y_train)
  return clf


# -

# ### Getting model evaluation metrics

# + pycharm={"name": "#%%\n"}
def eval_metrics(actual, pred):
  """Return a tuple containing model classification accuracy and confusion matrix.

  Keyword arguments:
  actual -- ndarray y_test containing true target values
  pred -- ndarray of the predicted target values by the model
  """
  accuracy = accuracy_score(actual, pred)
  conf_matrix = confusion_matrix(actual, pred)
  return accuracy, conf_matrix


# + pycharm={"name": "#%%\n"}
def get_model_evaluation_metrics(clf, X_test, y_test):
  """Return a tuple containing model classification accuracy and confusion matrix.

  Keyword arguments:
  clf -- classifier model
  X_test -- ndarray containing all test columns except target column
  y_test -- ndarray target column values to test the model
  """
  predicted_repayments = clf.predict(X_test)
  (accuracy, conf_matrix) = eval_metrics(y_test, predicted_repayments)
  return accuracy, conf_matrix


# -

# ### Tracking model on MLFLow

# + pycharm={"name": "#%%\n"}
def track_model_params(clf):
  """Log model params on MLFlow UI.

  Keyword arguments:
  clf -- classifier model
  """
  clf_params = clf.get_params()
  for param in clf_params:
      param_value = clf_params[param]
      mlflow.log_param(param, param_value)


# + pycharm={"name": "#%%\n"}
def track_model_metrics(clf, X_test, y_test):
  """Log model metrics on MLFlow UI.

  Keyword arguments:
  clf -- classifier model
  X_test -- ndarray containing all test columns except target column
  y_test -- ndarray target column values to test the model
  """
  (accuracy, conf_matrix) = get_model_evaluation_metrics(clf, X_test, y_test)
  mlflow.log_metric('accuracy', accuracy)
  #mlflow.log_metric('conf_matrix', conf_matrix)


# + pycharm={"name": "#%%\n"}
def track_model_version(clf):
  """Version model on MLFlow UI.

  Keyword arguments:
  clf -- classifier model
  """
  tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
  if tracking_url_type_store != 'file':
      mlflow.sklearn.log_model(clf, 'model', registered_model_name='RandomForestClassifier')
  else:
      mlflow.sklearn.log_model(clf, 'model')


# + pycharm={"name": "#%%\n"}
def set_mlflow_run_tags():
  """Set current MLFlow run tags."""
  tags = {'model_name': 'RandomForestClassifier'}
  mlflow.set_tags(tags)


# + pycharm={"name": "#%%\n"}
def train_and_track_model_in_mlflow():
  """Train model and track it with MLFLow"""
  (X_train, X_test, y_train, y_test) = get_split_train_data()
  logger = get_configured_logger()
  clf = train_random_forest_classifier(X_train, y_train)
  with mlflow.start_run():
    track_model_params(clf)
    track_model_metrics(clf, X_test, y_test)
    track_model_version(clf)
    set_mlflow_run_tags()


# + pycharm={"name": "#%%\n"}
if __name__ == '__main__':
    train_and_track_model_in_mlflow()