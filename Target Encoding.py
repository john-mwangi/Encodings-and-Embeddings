import copy #for deep copying
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import preprocessing #data preprocessing e.g. onehot, label encoding

import xgboost as xgb

# CUSTOM TARGET ENCODING
us_adults_raw = pd.read_csv("./adult.csv", na_values="?")

us_adults_raw.shape

us_adults_raw.head()

num_cols = us_adults_raw.select_dtypes(np.number).columns

us_adults_raw.income.value_counts()

#Remapping outcome variable
target_mapping = {"<=50K":0,">50K":1}

us_adults_raw.loc[:,"income"] = us_adults_raw.income.map(target_mapping)

us_adults_raw.head()

#Create training folds
from sklearn import model_selection

us_adults_raw["kfold"] = -1

us_adults_raw = us_adults_raw.sample(frac=1, random_state=123).reset_index(drop=True)

kf = model_selection.KFold(n_splits=5)

for fold,(trn_, val_) in enumerate(kf.split(X=us_adults_raw)):
    us_adults_raw.loc[val_,"kfold"]=fold

us_adults_raw.head()

us_adults_raw.kfold.value_counts()

features_raw = [
    f for f in us_adults_raw.columns if f not in ("kfold, income")
]

features_raw

for col in features_raw:
    if col not in num_cols:
        us_adults_raw.loc[:,col] = us_adults_raw[col].astype(str).fillna("NONE") #very important step


for col in features_raw:
    if col not in num_cols:
        lbl = preprocessing.LabelEncoder()
        
        lbl.fit(us_adults_raw[col])
        
        us_adults.loc[:,col] = lbl.transform(us_adults_raw[col])

us_adults.head()

#Target encoding
encoded_dfs = []

for fold in range(5):
    us_train = us_adults[us_adults.kfold != fold].reset_index(drop=True)
    us_test = us_adults[us_adults.kfold == fold].reset_index(drop=True)
    
    for column in features:
        mapping_dict = dict(
            us_train.groupby(column)["income"].mean()
        )
        
        us_test.loc[:,column + "_enc"] = us_test[column].map(mapping_dict)
    
    encoded_dfs.append(us_test)

encoded_df = pd.concat(encoded_dfs, axis=0)

encoded_df.head()

us_train.shape

us_test.shape

encoded_df.shape

encoded_df.loc[:,["age","age_enc","education","education_enc"]]


#Train using target encodings
for fold in range(5):
    df_train = encoded_df[encoded_df.kfold != fold]
    df_test = encoded_df[encoded_df.kfold == fold]

df_train.shape

df_test.shape

features = [
    f for f in encoded_df.columns if f not in ("income","kfold")
]

features

x_train = df_train[features]

x_test = df_test[features]

#help("xgboost.XGBClassifier")
model = xgb.XGBClassifier(max_depth = 7, n_jobs = -1)

model.fit(X=x_train, y=df_train.income)

valid_preds = model.predict_proba(X=x_test)[:,1]

metrics.roc_auc_score(df_test.income, valid_preds)


# SKLEARN TARGET ENCODING

#!pip install category_encoders
from category_encoders import TargetEncoder

us_adults = pd.read_csv("./adult.csv", na_values="?")

us_adults.head()

features_original = [
    f for f in us_adults.columns if f not in "income"
]

features_original

target_mapping

#Remap outcome variable
us_adults.loc[:,"income"] = us_adults.income.map(target_mapping)

us_adults.income.value_counts()

te = TargetEncoder(return_df=True, smoothing=0)

te.fit(X=us_adults[features_original], y=us_adults.income)

encoded_df_sk = te.transform(X=us_adults[features_original])

encoded_df_sk.shape

encoded_df_sk.head()
