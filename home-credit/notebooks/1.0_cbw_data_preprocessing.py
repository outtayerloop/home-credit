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

# + [markdown] id="rMOXnC8nBqF-"
# ## Importing the libraries

# + id="J44Q8wLcBs7W"
import pandas as pd
import numpy as np

# + [markdown] id="oYngGJuJABAl"
# ## Obtaining base train and test dataframes

# + [markdown] gradient={"editing": false} id="39d2ae9c-7e40-4baa-9c72-cdd5d912bb04"
# ### Creating train and test dataframes

# + gradient={"editing": false, "source_hidden": false} id="ddba72e6-af17-459d-b4e3-71fe3c6c7870"
train = pd.read_csv('./data/external/application_train.csv')

# + gradient={"editing": false, "source_hidden": false} id="52084b3b-2252-443f-a2e8-5d1dc2beb46c"
test = pd.read_csv('./data/external/application_test.csv')

# + [markdown] id="TrUG4j6iUHeW"
# ### Displaying dataframes first lines

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="szElCaNFUPY4" outputId="bdeada4f-3b52-4d03-a904-0ff7cf06d4ac"
train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="zXr269gaURb-" outputId="8ffb40c0-3acc-4bc2-9c27-da39ab295863"
test.head()

# + [markdown] id="0Oec65GiJ3Tp"
# ### Moving target to last column in train dataset

# + id="Gn1N4tAZJ6xn"
target_col = train.pop('TARGET')
train['TARGET'] = target_col

# + [markdown] id="zPF5c_yEHrGg"
# ### Dropping unused ID column

# + id="tSqMlcE7Hu75"
train = train.drop(['SK_ID_CURR'], axis=1)
test = test.drop(['SK_ID_CURR'], axis=1)

# + [markdown] id="x3iz2_FRjk4u"
# ### Organizing test set columns based on train set column order

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="oWUncy_bjzSu" outputId="3698ce1e-b413-49c9-d217-20ece83ee19f"
test = test[train.drop(['TARGET'], axis=1).columns]
test.head()

# + [markdown] id="RUGqhOrkF7VX"
# ## Taking care of missing data

# + colab={"base_uri": "https://localhost:8080/"} id="Av3ykPJYShqt" outputId="54370b58-29d6-414e-a6ae-ac560799f903"
train.shape

# + id="iw9XbJNVi7vC" colab={"base_uri": "https://localhost:8080/"} outputId="8682f9db-99e8-4839-c2f2-2e76520bbf46"
test.shape

# + id="m7aKDXPjF-C3" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="07b358d2-5a7a-4e89-b3e5-28d4f0c9dd7a"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_dtypes_train = train.dtypes[:-1]
num_cols_train = x_dtypes_train == np.number
X_train = train.iloc[:, :-1].values
imputer.fit(X_train[:, num_cols_train])
X_train[:, num_cols_train] = imputer.transform(X_train[:, num_cols_train])
train.iloc[:, :-1] = X_train
train.head()

# + id="bgQxbY5Mjef1" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="a0968ecb-a651-4e2a-b747-75bff7113a39"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_dtypes_test = test.dtypes
num_cols_test = x_dtypes_test == np.number
X_test = test.iloc[:, :].values
imputer.fit(X_test[:, num_cols_test])
X_test[:, num_cols_test] = imputer.transform(X_test[:, num_cols_test])
test.iloc[:, :] = X_test
test.head()

# + [markdown] id="hAvaeTqpRlxf"
# ### Get text features Na rows percentage

# + colab={"base_uri": "https://localhost:8080/"} id="9LuAtXAyO4sQ" outputId="3ac5960b-9d7d-4fa9-a16e-bee511d1c726"
na_cols_pctg_train = train[train.columns[train.isna().sum() > 0]].isna().sum() / train.shape[0]
na_cols_pctg_train

# + colab={"base_uri": "https://localhost:8080/"} id="8CQu3sKDj03U" outputId="363e0ab6-2746-40c1-8521-72c531a283b7"
na_cols_pctg_test = test[test.columns[test.isna().sum() > 0]].isna().sum() / test.shape[0]
na_cols_pctg_test

# + [markdown] id="wFdGMgFhRt8R"
# ### Drop text features Na rows

# + id="nxIcZ8luRwAC"
train = train.dropna(axis=0)

# + colab={"base_uri": "https://localhost:8080/"} id="meGsOmtGSDSS" outputId="c1563ce1-b450-475c-abb1-f8b9b7f49280"
train.shape

# + id="nxZK5Xi-kQM9"
test = test.dropna(axis=0)

# + id="kBewfq1NkUay" colab={"base_uri": "https://localhost:8080/"} outputId="81bb2cd2-956a-4e9d-a9c9-562026d321b0"
test.shape

# + [markdown] id="w9hpg1ZXAxEn"
# ## Encoding categorical data

# + [markdown] id="mtbCfv0RBzMe"
# ### Encoding the independent variables

# + id="jhSb0ulEDB3f" colab={"base_uri": "https://localhost:8080/"} outputId="694c90dc-049e-4220-a0ee-7234c55bd6c4"
# Get textual columns indexes
txt_cols_train = train.select_dtypes('object').columns
txt_indexes_train = train.columns.get_indexer(txt_cols_train)
txt_indexes_train

# + id="0-VffTCdkkWL" colab={"base_uri": "https://localhost:8080/"} outputId="080362f5-2ed5-4d84-9f7d-288bd5ee1069"
txt_cols_test = test.select_dtypes('object').columns
txt_indexes_test = test.columns.get_indexer(txt_cols_test)
txt_indexes_test

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="7K1WiR32BmU2" outputId="e8769065-c425-4abe-fb0d-547b105f5cf5"
from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
X_train = train.iloc[:, :-1].values
for i in txt_indexes_train:
  X_train[:, i] = label_encoder_x.fit_transform(X_train[:, i])
train.iloc[:, :-1] = X_train
train.head()

# + id="GpYM3xmVkz3j" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="e8250a65-9692-4869-d366-3a36a172c3ee"
from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
X_test = test.iloc[:, :].values
for i in txt_indexes_test:
  X_test[:, i] = label_encoder_x.fit_transform(X_test[:, i])
test.iloc[:, :] = X_test
test.head()

# + [markdown] id="Vmv4pyCCgbh5"
# ### Encoding the dependent variable

# + [markdown] id="Z_3KZvBsgqtR"
# #### Checking target possible values

# + colab={"base_uri": "https://localhost:8080/"} id="4KKTTxPtgr-6" outputId="11b0565e-d692-466d-cb06-b6d1ff0b2318"
train['TARGET'].unique()

# + [markdown] id="-DevuIXwgzZg"
# The categorical target variable can only take two values : **0** and **1**, thus no further encoding is required

# + [markdown] id="x_yhicg4hlkl"
# ## Feature scaling

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="Mym1cklwhoAT" outputId="2425444c-f324-4083-9c6a-fa77ea36fd44"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # standardization implies values between approximately -3 and 3
X_train = train.iloc[:, :-1].values
X_train[:, num_cols_train] = sc.fit_transform(X_train[:, num_cols_train]) # we don't standardize encoded textual dimensions.
train.iloc[:, :-1] = X_train
train.head()

# + id="lLoMGwhnlvD5" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="f61e7474-002d-4d5d-a9e1-f004ccd4a8a8"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # standardization implies values between approximately -3 and 3
X_test = test.iloc[:, :].values
X_test[:, num_cols_test] = sc.fit_transform(X_test[:, num_cols_test]) # we don't standardize encoded textual dimensions.
test.iloc[:, :] = X_test
test.head()

# + [markdown] id="To5vZ44roDjL"
# ## Feature selection

# + [markdown] id="E6PwqzS3L6fp"
# ### Removing features with at least 50% Na values (percentage computed from train set)

# + colab={"base_uri": "https://localhost:8080/"} id="pDC6VeFGMAbZ" outputId="9a6c1744-30aa-44aa-e3c8-37b9828b8d97"
dropped_cols = na_cols_pctg_train[na_cols_pctg_train >= 0.5].axes[0].tolist()
train = train.drop(dropped_cols, axis=1)
train.shape

# + colab={"base_uri": "https://localhost:8080/"} id="_baEUNBMQNTL" outputId="1b05c710-45a7-4e62-a3de-06cf4d8830c4"
test = test.drop(dropped_cols, axis=1)
test.shape

# + [markdown] id="YocDxkHkQUZh"
# ### Removing features with a modality that appears with a probability of at least 80%

# + colab={"base_uri": "https://localhost:8080/"} id="vYuZbotJQXsJ" outputId="183a416d-a001-43fe-db57-940198d74d31"
PROBABILITY_THRESHOLD = 0.8
train_without_target = train.drop('TARGET', axis=1)
cols_train = train_without_target.columns.tolist()
cols_to_drop_train = []
for col in cols_train:
  mods_pctg = train_without_target[col].value_counts() / train_without_target[col].value_counts().sum()
  for pctg in mods_pctg:
    if pctg >= PROBABILITY_THRESHOLD:
      cols_to_drop_train.append(col)
train = train.drop(cols_to_drop_train, axis=1)
train.shape

# + colab={"base_uri": "https://localhost:8080/"} id="RlXY3Vf06ccQ" outputId="f12a18c0-c2c8-4c75-8384-a10efee87608"
test = test.drop(cols_to_drop_train, axis=1)
test.shape

# + [markdown] id="4dvUPZ3vfs5L"
# ### Performing PCA

# + id="jAqEB435kKN1"
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt

#train_array = train.drop('TARGET',axis=1).values
#pca = PCA().fit(train_array)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')

# + id="J3RyDbkmfu2f"
#pipe = Pipeline([('scaler', StandardScaler()), ('acp', PCA(n_components=5))])
#train_pca = pipe.fit_transform(train_array)
#train_pca.shape

# + [markdown] id="GrF7z1TFpxeN"
# ### Removing features with a positive / negative correlation of at least 80%

# + id="U335_phXqkfb"
#train_cols = train.columns
#correlation = train.corr(method="pearson")
#target_corrs = correlation['TARGET']
#[corr for corr in target_corrs if corr >= 0.2 or corr <= -0.1]


# List of new correlations
#new_corrs = []

# Iterate through the columns 
#for col in train_cols:
    # Calculate correlation with the target
#    corr = train['TARGET'].corr(train[col])
    # Append the list as a tuple
#    new_corrs.append(corr)
#[corr for corr in new_corrs if corr >= 0.1 or corr <= -0.1]

# + [markdown] id="-ag0L1nB6UsZ"
# ## Preprocessed data export to CSV

# + id="G2iGO7OO6X65"
train.to_csv('./data/processed/preprocessed_application_train.csv', index=False)
test.to_csv('./data/processed/preprocessed_application_test.csv', index=False)
