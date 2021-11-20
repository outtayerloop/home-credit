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
# ## Calculating Shapley values for fit XGBoost model

# + id="wo3KoVTlYMRG" pycharm={"name": "#%%\n"}
def get_fit_model_shapley_values_and_explainer(fit_xgb_model):
  """Return a tuple with a list containing computed Shapley values from fit XGBoost model
  and the obtained TreeExplainer.

  Keyword arguments:
  fit_xgb_model -- Fitted XGBoost model
  """
  explainer = shap.TreeExplainer(fit_xgb_model)
  data_for_prediction = pd.read_csv('./data/processed/processed_application_test.csv')
  shap_values = explainer.shap_values(data_for_prediction)
  return shap_values, explainer


# -

# ## Vizualizing explanations for a single line in test dataset

# + pycharm={"name": "#%%\n"}
def vizualize_single_line_xai(explainer, line, line_index, shap_values):
  """Call SHAP force_plot to display explanations for a single line in test dataset.

  Keyword arguments:
  explainer -- SHAP TreeExplainer obtained from fit XGBoost model
  line -- single line in test dataset that will be used for prediction
  line_index -- Test dataset line index
  shap_values -- Computed SHAP values from XGBoost fit model
  """
  shap.initjs()
  shap.force_plot(explainer.expected_value[line_index], shap_values[line_index], line)
