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


# + [markdown] id="uHnlnHIjZtGG"
# ## Calculating Shapley values for fitted XGBoost model

# + id="wo3KoVTlYMRG"
def get_fit_model_shapley_values(fitted_xgb_model):
  """Return a list containing computed Shapley values from fitted XGBoost model.

  Keyword arguments:
  fitted_xgb_model -- Fitted XGBoost model
  """
  explainer = shap.TreeExplainer(fitted_xgb_model)
  shap_values = explainer.shap_values(data_for_prediction)
  return shap_values


# + [markdown] id="nwKuEIeihzsa"
# ## Vizualizing explanations for a single line in test dataset (without target column)

# + id="n-mk7pL0hytI"
def vizualize_single_line_xai(line, line_index):
  """Return a list containing computed Shapley values from fitted XGBoost model.

  Keyword arguments:
  line -- single line in test dataset that will be used for prediction
  line_index -- Line index
  """
  shap.initjs()
  line_array = line.values.reshape(1, -1)
  shap.force_plot(explainer.expected_value[line_index], shap_values[line_index], line)


# + [markdown] id="RA_Eqwsr6avD"
# ## Vizualizing explanations for all lines at once in test dataset (without target column)

# + id="Gruyw1Ke6gU3"
def vizualize_full_lines_xai(lines):
  """Return a list containing computed Shapley values from fitted XGBoost model.

  Keyword arguments:
  lines -- all lines in test dataset that will be used for prediction
  """
  shap.initjs()
  line_array = line.values.reshape(1, -1)
  shap.force_plot(explainer.expected_value[line_index], shap_values[line_index], line)


# + [markdown] id="9Fty4H7CA8K_"
# ## Vizualizing summary plot for each class on the whole dataset with fitted XGBoost model

# + id="5deJ6XdqBBzb"
def vizualize_shap_summary_plot(fitted_xgb_model, lines)
  """Vizualize summary plot for each class on the whole dataset with fitted XGBoost model.

    Keyword arguments:
    fitted_xgb_model -- Fitted XGBoost model
    lines -- all lines in test dataset that will be used for prediction
  """
  explainer = shap.TreeExplainer(fitted_xgb_model)
  shap_values = explainer.shap_values(lines)
  shap.summary_plot(shap_values[1],lines)
