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

# + [markdown] id="7bC2CZMQ793n"
# # Gradient Boosting model train

# + [markdown] id="YuYqZHHj2pwk"
# - https://www.kaggle.com/beagle01/prediction-with-gradient-boosting-classifier

# + [markdown] id="fHwQSvj17-vR"
# ## Importing the libraries

# + id="ksABrZmK8tWx"

import pandas as pd

# + [markdown] id="5HFfVPgE8FTy"
# ## Importing the dataset

# + id="vze1y0gl8rXd"
train = pd.read_csv('../../../../data/processed/preprocessed_application_train.csv')
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

# + colab={"base_uri": "https://localhost:8080/"} id="IGUl82S04Q3t" outputId="a2adbb38-f263-419e-fe92-fe6884f3e253"
train.shape

# + id="CiDl6Nz0u_KG"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# + [markdown] id="gls4kHR49jZ3"
# ## Training the Gradient Boosting model on the training set

# + id="bwIDRnnz9nz2"
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf = clf.fit(X_train, y_train)

# + [markdown] id="OGC0rTyXp91A"
# ## Adding MLFLow workflow

# + id="BC1wC_0irJXW"
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    conf_matrix = confusion_matrix(actual, pred)
    return accuracy, conf_matrix


# + id="-YuodLe9qB1Y"
import logging
import mlflow
from urllib.parse import urlparse

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

with mlflow.start_run():
  clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
  clf = clf.fit(X_train, y_train)
  predicted_repayments = clf.predict(X_test)

  (accuracy, conf_matrix) = eval_metrics(y_test, predicted_repayments)

  clf_params = clf.get_params()

  for param in clf_params:
    print(param)

  mlflow.log_metric('accuracy', accuracy)
  #mlflow.log_metric('conf_matrix', conf_matrix)

  mlflow_tracking_uri = mlflow.get_tracking_uri()
  print(mlflow_tracking_uri)

  tracking_url_type_store = urlparse(mlflow_tracking_uri).scheme

  # Model registry does not work with file store
  if tracking_url_type_store != 'file':

      # Register the model
      # There are other ways to use the Model Registry, which depends on the use case,
      # please refer to the doc for more information:
      # https://mlflow.org/docs/latest/model-registry.html#api-workflow
      mlflow.sklearn.log_model(clf, 'model', registered_model_name='GradientBoostingClassifier')
  else:
      mlflow.sklearn.log_model(clf, 'model')
