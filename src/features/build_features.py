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

# + [markdown] gradient={"editing": false} id="1a069376-2802-477e-ae22-8192da45be11"
# # Data preprocessing

# + id="J44Q8wLcBs7W"
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.data import make_dataset as md


# + [markdown] id="oYngGJuJABAl"
# ## Obtaining base train and test dataframes

# + [markdown] gradient={"editing": false} id="39d2ae9c-7e40-4baa-9c72-cdd5d912bb04"
# ### Creating train and test dataframes

# + id="iFDLQubNVQtP"
def get_dataframes():
    """Return a tuple containing train and test dataframes."""
    md.retrieve_data()
    train = pd.read_csv('./data/external/application_train.csv')
    test = pd.read_csv('./data/external/application_test.csv')
    return train, test


# + [markdown] id="0Oec65GiJ3Tp"
# ### Moving target to last column in train dataset

# + id="Gn1N4tAZJ6xn"
def position_target_column(train):
    """Return train dataframe with target as last column.

  Keyword arguments:
  train -- the train dataframe
  """
    target_col = train.pop('TARGET')
    train['TARGET'] = target_col
    return train


# + [markdown] id="zPF5c_yEHrGg"
# ### Dropping unused ID column

# + id="tSqMlcE7Hu75"
def drop_id_column(train, test):
    """Return a tuple containing train and test dataframes without id column.

  Keyword arguments:
  train -- the train dataframe
  test -- the test dataframe
  """
    train = train.drop(['SK_ID_CURR'], axis=1)
    test = test.drop(['SK_ID_CURR'], axis=1)
    return train, test


# + [markdown] id="x3iz2_FRjk4u"
# ### Organizing test set columns based on train set column order

# + id="oWUncy_bjzSu"
def reorder_test_columns(train, test):
    """Return test dataframe with columns organized following train dataframe columns order.

  Keyword arguments:
  train -- the train dataframe
  test -- the test dataframe
  """
    test = test[train.drop(['TARGET'], axis=1).columns]
    return test


# + [markdown] id="RUGqhOrkF7VX"
# ## Taking care of missing data

# + id="m7aKDXPjF-C3"
def impute_train_missing_data(train):
    """
  Return tuple containing train dataframe with median imputed in place of missing numerical values
  and a Series with its numerical columns.

  Keyword arguments:
  train -- the train dataframe
  """
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x_dtypes_train = train.dtypes[:-1]
    num_cols_train = x_dtypes_train == np.number
    X_train = train.iloc[:, :-1].values
    imputer.fit(X_train[:, num_cols_train])
    X_train[:, num_cols_train] = imputer.transform(X_train[:, num_cols_train])
    train.iloc[:, :-1] = X_train
    return train, num_cols_train


# + id="bgQxbY5Mjef1"
def impute_test_missing_data(test):
    """
  Return tuple containing test dataframe with median imputed in place of missing numerical values
  and a Series with its numerical columns.

    Keyword arguments:
    test -- the test dataframe
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x_dtypes_test = test.dtypes
    num_cols_test = x_dtypes_test == np.number
    X_test = test.iloc[:, :].values
    imputer.fit(X_test[:, num_cols_test])
    X_test[:, num_cols_test] = imputer.transform(X_test[:, num_cols_test])
    test.iloc[:, :] = X_test
    return test, num_cols_test


# + [markdown] id="hAvaeTqpRlxf"
# ### Getting text features Na rows percentage

# + id="9LuAtXAyO4sQ"
def get_train_na_percentages(train):
    """
  Return a Series with the percentage of Na values per columns in train dataframe.
  Must be called just after impute_train_missing_data().

  Keyword arguments:
  train -- the train dataframe
  """
    na_cols_pctg_train = train[train.columns[train.isna().sum() > 0]].isna().sum() / train.shape[0]
    return na_cols_pctg_train


# + [markdown] id="wFdGMgFhRt8R"
# ### Dropping text features Na rows

# + id="nxIcZ8luRwAC"
def drop_textual_feat_na_rows(train, test):
    """Return a tuple containing train and test dataframes without textual features Na rows.

  Keyword arguments:
  train -- the train dataframe
  test -- the test dataframe
  """
    train = train.dropna(axis=0)
    test = test.dropna(axis=0)
    return train, test


# + [markdown] id="w9hpg1ZXAxEn"
# ## Encoding categorical data

# + [markdown] id="mtbCfv0RBzMe"
# ### Encoding the independent variables

# + id="jhSb0ulEDB3f"
def get_textual_column_indexes(train, test):
    """Return a tuple containing an ndarray with train and test textual column indexes.

  Keyword arguments:
  train -- the train dataframe
  test -- the test dataframe
  """
    txt_cols_train = train.select_dtypes('object').columns
    txt_indexes_train = train.columns.get_indexer(txt_cols_train)
    txt_cols_test = test.select_dtypes('object').columns
    txt_indexes_test = test.columns.get_indexer(txt_cols_test)
    return txt_indexes_train, txt_indexes_test


# + id="7K1WiR32BmU2"
def label_encode_train(train, txt_indexes_train):
    """Return the train dataframe with label-encoded textual features.

  Keyword arguments:
  train -- the train dataframe
  txt_indexes_train -- ndarray of train textual column indexes
  """
    label_encoder_x = LabelEncoder()
    X_train = train.iloc[:, :-1].values
    for i in txt_indexes_train:
        X_train[:, i] = label_encoder_x.fit_transform(X_train[:, i])
    train.iloc[:, :-1] = X_train
    return train


# + id="GpYM3xmVkz3j"
def label_encode_test(test, txt_indexes_test):
    """Return the test dataframe with label-encoded textual features.

  Keyword arguments:
  test -- the test dataframe
  txt_indexes_test -- ndarray of test textual column indexes
  """
    label_encoder_x = LabelEncoder()
    X_test = test.iloc[:, :].values
    for i in txt_indexes_test:
        X_test[:, i] = label_encoder_x.fit_transform(X_test[:, i])
    test.iloc[:, :] = X_test
    return test


# + id="eUpCP9FC42fJ"
def one_hot_encode_train(train, txt_indexes_train):
  """Return the train dataframe with one-hot-encoded textual features.

  Keyword arguments:
  train -- the train dataframe
  txt_indexes_train -- ndarray of train textual column indexes
  """
  train_dummies = pd.get_dummies(train.iloc[:, txt_indexes_train])
  train.drop(train.select_dtypes('object').columns, axis=1, inplace=True)
  train = pd.concat([train, train_dummies], axis=1)
  train = position_target_column(train)
  return train


def one_hot_encode_test(test, txt_indexes_test):
  """Return the test dataframe with label-encoded textual features.

  Keyword arguments:
  test -- the test dataframe
  txt_indexes_test -- ndarray of test textual column indexes
  """
  test_dummies = pd.get_dummies(test.iloc[:, txt_indexes_test])
  test.drop(test.select_dtypes('object').columns, axis=1, inplace=True)
  test = pd.concat([test, test_dummies], axis=1)
  return test


def align_encoded_train_test(train, test):
    """Return encoded train and test dataframes with same columns (except target that is still in train).

      Keyword arguments:
      test -- the test dataframe
      train -- the train dataframe
      """
    # Align the training and testing data, keep only columns present in both dataframes
    target_col = train['TARGET']
    train, test = train.align(test, join='inner', axis=1)
    train = pd.concat([train, target_col], axis=1)
    return train, test


# + [markdown] id="x_yhicg4hlkl"
# ## Feature scaling

# + id="Mym1cklwhoAT"
def standardize_train(train, num_cols_train):
    """Return the train dataframe with standardized numerical features (not the encoded textual dimensions).

  Keyword arguments:
  train -- the train dataframe
  """
    sc = StandardScaler()
    X_train = train.iloc[:, :-1].values
    X_train[:, num_cols_train] = sc.fit_transform(X_train[:, num_cols_train])
    train.iloc[:, :-1] = X_train
    return train


# + id="lLoMGwhnlvD5"
def standardize_test(test, num_cols_test):
    """Return the test dataframe with standardized numerical features (not the encoded textual dimensions).

  Keyword arguments:
  test -- the test dataframe
  """
    sc = StandardScaler()  # standardization implies values between approximately -3 and 3
    X_test = test.iloc[:, :].values
    X_test[:, num_cols_test] = sc.fit_transform(
        X_test[:, num_cols_test])  # we don't standardize encoded textual dimensions.
    test.iloc[:, :] = X_test
    return test


# + [markdown] id="-ag0L1nB6UsZ"
# ## Exporting preprocessed data to CSV files

# + id="G2iGO7OO6X65"
def export_dataframes_to_csv_files(train, test):
    """Export train and test dataframes to CSV files to ./data/processed path.

  Keyword arguments:
  train -- the train dataframe
  test -- the test dataframe
  """
    train.to_csv('./data/processed/processed_application_train.csv', index=False)
    test.to_csv('./data/processed/processed_application_test.csv', index=False)


# + [markdown] id="oELQf611CPH7"
# ## Executing the preprocessing workflow

# + id="TCTVgKVsCVsR" pycharm={"name": "#%%\n"}
def build_features():
    """Build feature and export train and test dataframes to CSVs"""
    (train, test) = get_dataframes()
    train = position_target_column(train)
    (train, test) = drop_id_column(train, test)
    test = reorder_test_columns(train, test)
    (train, num_cols_train) = impute_train_missing_data(train)
    (test, num_cols_test) = impute_test_missing_data(test)
    na_cols_pctg_train = get_train_na_percentages(train)
    (train, test) = drop_textual_feat_na_rows(train, test)
    train = standardize_train(train, num_cols_train)
    test = standardize_test(test, num_cols_test)
    (txt_indexes_train, txt_indexes_test) = get_textual_column_indexes(train, test)
    train = one_hot_encode_train(train, txt_indexes_train)
    test = one_hot_encode_test(test, txt_indexes_test)
    train, test = align_encoded_train_test(train, test)
    export_dataframes_to_csv_files(train, test)