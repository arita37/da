#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


### Install Requirement
get_ipython().system("pip install -r requirements.txt")


# In[106]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "")
get_ipython().run_line_magic("matplotlib", "inline")


import gc
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mlens.ensemble import BlendEnsemble, SuperLearner
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

####Ensemble Learning
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm_notebook
from util_feature import *
from util_model import *

warnings.filterwarnings("ignore")


# In[81]:


print("ok")


# In[69]:


folder = os.getcwd() + "/"


# In[70]:


df = pd.read_csv(folder + "/data/address_matching_data.csv")
df.head(5)


# In[20]:


df.describe()


# In[21]:


df.columns, df.dtypes


# In[ ]:


# In[22]:


colid = "id"
colnum = [
    "name_levenshtein_simple",
    "name_trigram_simple",
    "name_levenshtein_term",
    "name_trigram_term",
    "city_levenshtein_simple",
    "city_trigram_simple",
    "city_levenshtein_term",
    "city_trigram_term",
    "zip_levenshtein_simple",
    "zip_trigram_simple",
    "zip_levenshtein_term",
    "zip_trigram_term",
    "street_levenshtein_simple",
    "street_trigram_simple",
    "street_levenshtein_term",
    "street_trigram_term",
    "website_levenshtein_simple",
    "website_trigram_simple",
    "website_levenshtein_term",
    "website_trigram_term",
    "phone_levenshtein",
    "phone_trigram",
    "fax_levenshtein",
    "fax_trigram",
    "street_number_levenshtein",
    "street_number_trigram",
]

colcat = ["phone_equality", "fax_equality", "street_number_equality"]
coltext = []

coly = "is_match"


# In[75]:


# Normalize to NA
df = df.replace("?", np.nan)


# In[76]:


### colnum procesing
for x in colnum:
    df[x] = df[x].astype("float32")

print(df.dtypes)


# In[71]:


##### Colcat processing  :^be caregul thant test contain same category
colcat_map = pd_colcat_mapping(df, colcat)

for col in colcat:
    df[col] = df[col].apply(lambda x: colcat_map["cat_map"][col].get(x))

print(df[colcat].dtypes, colcat_map)


# In[74]:


# In[72]:


#### ColTarget Distribution
coly_stat = pd_stat_distribution(df[["id", coly]], subsample_ratio=1.0)
coly_stat


# In[ ]:


# In[ ]:


# In[77]:


#### Col numerics distribution
colnum_stat = pd_stat_distribution(df[colnum], subsample_ratio=0.6)
colnum_stat


# In[ ]:


# In[78]:


#### Col stats distribution
colcat_stat = pd_stat_distribution(df[colcat], subsample_ratio=0.3)
colcat_stat


# In[30]:


### BAcKUP data before Pre-processing

dfref = copy.deepcopy(df)
print(dfref.shape)


# In[145]:


df, colnum_map = pd_colnum_tocat(
    df, colname=colnum, colexclude=None, bins=5, suffix="_bin", method=""
)


print(colnum_map)


# In[85]:


colnum_bin = list(colnum_map.keys())
print(colnum_bin)


# In[86]:


dfnum_hot = pd_col_to_onehot(df[colnum_bin], colname=colnum_bin, returncol=0)


# In[153]:


colnum_hot = list(dfnum_hot.columns)
dfnum_hot.head(10)


# In[126]:


dfcat_hot = pd_col_to_onehot(df[colcat], colname=colcat, returncol=0)
colcat_hot = list(dfcat_hot.columns)
dfcat_hot.head(5)


# In[ ]:


# In[89]:


#### Train
X, yy = pd.concat((dfnum_hot, dfcat_hot), axis=1).values, df[coly].values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, random_state=42, test_size=0.5, shuffle=True)


print(Xtrain.shape, Xtest.shape)


# In[ ]:


# In[114]:


Xtrain


# In[127]:


# In[ ]:


# In[90]:


### L1 penalty to reduce overfitting
clf_log = sk.linear_model.LogisticRegression(penalty="l2", class_weight="balanced")


# In[91]:


clf_log, dd = sk_model_eval_classification(clf_log, 1, Xtrain, ytrain, Xtest, ytest)


# In[92]:


sk_model_eval_classification_cv(clf_log, X, yy, test_size=0.5, ncv=3)


# In[93]:


colall = list(dfnum_hot.columns) + list(dfcat_hot.columns)

clf_log_feat = sk_feature_impt_logis(clf_log, colall)
clf_log_feat


# In[208]:


# In[44]:


1


# In[96]:


### Light GBM
clf_lgb = lgb.LGBMClassifier(
    learning_rate=0.125,
    metric="l2",
    max_depth=15,
    n_estimators=50,
    objective="binary",
    num_leaves=38,
    njobs=-1,
)


# In[97]:


clf_lgb, dd_lgb = sk_model_eval_classification(clf_lgb, 1, Xtrain, ytrain, Xtest, ytest)


# In[98]:


shap.initjs()

dftest = pd.DataFrame(columns=colall, data=Xtest)

explainer = shap.TreeExplainer(clf_lgb)
shap_values = explainer.shap_values(dftest)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0, :], dftest.iloc[0, :])


# In[ ]:


# visualize the training set predictions
# shap.force_plot(explainer.expected_value, shap_values, dftest)

# Plot summary_plot as barplot:
# shap.summary_plot(shap_values, Xtest, plot_type='bar')


# In[112]:


lgb_feature_imp = pd.DataFrame(
    sorted(zip(clf_lgb.feature_importances_, colall)), columns=["value", "feature"]
)
lgb_feature_imp = lgb_feature_imp.sort_values("value", ascending=0)
print(lgb_feature_imp)


plotbar(
    lgb_feature_imp.iloc[:10, :],
    colname=["value", "feature"],
    title="feature importance",
    savefile="lgb_feature_imp.png",
)


# In[118]:


kf = StratifiedKFold(n_splits=3, shuffle=True)
# partially based on https://www.kaggle.com/c0conuts/xgb-k-folds-fastai-pca
clf_list = []
for itrain, itest in kf.split(X, yy):
    print("###")
    Xtrain, Xval = X[itrain], X[itest]
    ytrain, yval = yy[itrain], yy[itest]
    clf_lgb.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], early_stopping_rounds=20)

    clf_list.append(clf_lgb)


# In[122]:


for i, clfi in enumerate(clf_list):
    print(i)
    clf_lgbi, dd_lgbi = sk_model_eval_classification(clfi, 0, Xtrain, ytrain, Xtest, ytest)


# In[ ]:


# In[49]:


# In[ ]:


# In[123]:


# Fitting a SVM
clf_svc = SVC(C=1.0, probability=True)  # since we need probabilities

clf_svc, dd_svc = sk_model_eval_classification(clf_svc, 1, Xtrain, ytrain, Xtest, ytest)


# In[228]:


# In[231]:


# # Ensembling

# In[ ]:


# In[54]:


estimators = [("clf_log", clf_log), ("clf_lgb", clf_lgb), ("clf_svc", clf_svc)]

clf_ens1 = VotingClassifier(estimators, voting="soft")  # Soft is required

print(clf_ens1)


# In[55]:


sk_model_eval_classification(clf_ens1, 1, Xtrain, ytrain, Xtest, ytest)


# In[ ]:


# In[ ]:


# In[ ]:


# # Predict values

# In[129]:


dft = pd.read_csv(folder + "/data/address_matching_data.csv")


# In[130]:


#####
dft = dft.replace("?", np.nan)


# In[131]:


dft[colcat].head(3)


# In[132]:


#### Pre-processing  cat :  New Cat are discard, Missing one are included
for col in colcat:
    try:
        dft[col] = dft[col].apply(lambda x: colcat_map["cat_map"][col].get(x))
    except Exception as e:
        print(col, e)


dft_colcat_hot = pd_col_to_onehot(dft[colcat], colcat)


for x in colcat_hot:
    if not x in dft_colcat_hot.columns:
        dft_colcat_hot[x] = 0
        print(x, "added")


dft_colcat_hot[colcat_hot].head(5)


# In[133]:


dft_colcat_hot.head(4)


# In[151]:


#### Pre-processing num : are discard, Missing one are included

dft_numbin, _ = pd_colnum_tocat(
    dft[colnum],
    colname=colnum,
    colexclude=None,
    colbinmap=colnum_map,
    bins=0,
    suffix="_bin",
    method="",
)


# In[157]:


dft_numbin.head(5)


# In[167]:


dft_num_hot = pd_col_to_onehot(dft_numbin[colnum_bin], colname=colnum_bin, colonehot=colnum_hot)


# In[165]:


dft_num_hot.head(5)


# In[168]:


print(dft_num_hot.shape, dfnum_hot.shape)


# In[161]:


# In[169]:


#### Train
X = pd.concat((dft_num_hot, dfcat_hot), axis=1).values

print(X.shape)


# In[175]:


dft[coly] = clf_ens1.predict(X)


# In[ ]:


# In[176]:


dft.head(5)


# In[177]:


dft.groupby(coly).agg({"id": "count"})


# In[172]:


# In[183]:


dft[["id", "is_match"]].to_csv("adress_pred.csv", index=False, mode="w")


# In[ ]:


# In[ ]:


# In[ ]:


# In[174]:


# In[ ]:


# In[ ]:


# ensemble = SuperLearner(scorer=roc_auc_score, random_state=32, verbose=2)


def model_ensemble_build(clf_list, proba, **kwargs):
    """Return an ensemble."""
    ensemble = BlendEnsemble(**kwargs)
    ensemble.add(clf_list, proba=proba)  # Specify 'proba' here
    ensemble.add_meta(LogisticRegression())
    return ensemble


clf_list = [clf_log, clf_lgb]

clf_ens = model_ensemble_build(clf_list, proba=True, scorer=roc_auc_score)


print(clf_ens)


# In[ ]:


# In[ ]:


train_df.loc[train_df["is_match"] == -1, "is_match"] = 0

test_df = pd.read_csv("./data/address_matching_test.csv")
test_df["is_match"] = np.nan


# In[33]:


# In[34]:


### NA handling
df_ = train_df.append(test_df)
use_columns = [s for s in df_.columns if s not in ["id", "is_match"]]
df = df_[use_columns]

for c in df.columns:
    if df[c].dtype == "object":
        df.loc[df[c] == "?", c] = 0
    else:
        print("skip ", c)


# In[35]:


### Encode numerical into Category to handle NA distribution


# In[179]:


0


# In[38]:


# In[39]:


# In[16]:


# In[ ]:


# In[40]:


# In[41]:


# In[ ]:


# In[ ]:


# In[180]:


0


# In[ ]:


"""
Sparse Logistics


"""


# In[181]:


0


# In[46]:


### NO Null Features
len(df_featlogis[df_featlogis["coef_abs"] > 0.0])


# In[ ]:


# In[ ]:


# In[ ]:


preds = clf.predict(test_df[feats])
preds[preds == 0] = -1


# In[ ]:


test_df = pd.read_csv("./data/address_matching_test.csv")
test_df["is_match"] = preds.astype(int)
test_df = test_df.reset_index()
test_df[["id", "is_match"]].to_csv("result.csv", index=False)
