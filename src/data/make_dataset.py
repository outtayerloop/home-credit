# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Data retrieval from Kaggle API

# + pycharm={"name": "#%%\n"}
import subprocess


# + pycharm={"name": "#%%\n"}
# Works on Windows machines
def retrieve_data():
    """Remove contents from external data folder and download relevant Kaggle competition CSV files into it."""
    subprocess.run(['powershell', '-Command', 'Remove-Item ./data/external/* -Recurse -Force'])
    subprocess.run('kaggle competitions download -c home-credit-default-risk -p ./data/external/')
    subprocess.run(['powershell', '-Command',
                    'Expand-Archive ./data/external/home-credit-default-risk.zip -DestinationPath ./data/external/'])
    subprocess.run(['powershell', '-Command', 'rm ./data/external/home-credit-default-risk.zip'])
