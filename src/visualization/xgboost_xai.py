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

# + [markdown] id="YBrOq-o7aUzi"
# # XGBoost eXplainable AI

# + id="GYY02WszYLli"
import shap
import pandas as pd


# + [markdown] id="uHnlnHIjZtGG"
# ## Calculating Shapley values for fitted XGBoost model

# + id="wo3KoVTlYMRG" pycharm={"name": "#%%\n"}
def get_fitted_model_shapley_values(fitted_xgb_model):
  """Return a list containing computed Shapley values from fitted XGBoost model.

  Keyword arguments:
  fitted_xgb_model -- Fitted XGBoost model
  """
  explainer = shap.TreeExplainer(fitted_xgb_model)
  data_for_prediction = pd.read_csv('./../../../data/processed/processed_application_test.csv')
  shap_values = explainer.shap_values(data_for_prediction)
  return shap_values
