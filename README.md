home-credit
==============================

applications of big data project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
--------
## Project presentation

Home-credit is an application aims to:

- Create and validate 3 Machine learning models (XGboost, Random Forest, Gradient Boosting)
  - Preparation: collecting, cleaning data
  - Feature engineering
  - Training and Validating data
  - Prediction data

- Provide client repayment abilities prediction, by selection one of the 3 Machine Learning models trained before


| Algorithm Name    | Aim                                                                                                                                                                                                                       | 
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Random forest     | supervised machine learning algorithm that is constructed from decision tree algorithms                                                                                                                                   |
| XGboost           | optimized distributed gradient boosting library designed to be highly efficient, flexible and portable                                                                                                                    |
| Gradient boosting | method standing out for its prediction speed and accuracy, particularly with large and complex datasets. From Kaggle competitions to machine learning solutions for business, this algorithm has produced the best results|

*XGboost and Gradient boosting are one type of Boosting algorithm which is a supervised machine learning and consists of an ensemble learning technique that uses a set of Machine Learning algorithms to convert weak learner to strong learners in order to increase the accuracy of the model.*

--------
## Installation


Prerequisites:

> Python 3.X (Programming language used in this application)

> Poetry (Python package manager), with following command: 
>
>> "(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -"

> Pip (Python Package installer), after downloaded get-pip.py move to the folder where get-pip.py is and run: **python get-pip.py**

1. Clone the following project repository: https://github.com/wiwiii/home-credit
2. Install global dependencies
    - From top-level directory folder (home-credit) type: **pip install -r requirements.txt**
3. Install application dependencies
    - From top-level directory folder (home-credit) type: **poetry install**

--------
## Getting started

Command based on what you want to achieve


| Goal                                | Command                                                      | 
|-------------------------------------|--------------------------------------------------------------|
| Home credit default risk prediction | Move to top-level directory and run **python app_predict.py**| 
| Create and validate ML models       | Move to top-level directory and run **python app_train.py**  |       
--------

## Contributors

Wiem **CHOUCHANE**

Carine **TALENDIER**

Brunelle **MALANDILA LEYA**


Git repository: https://github.com/wiwiii/home-credit

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
