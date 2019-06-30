
# coding: utf-8

# In[1]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings('ignore')


# In[2]:


train_df = pd.read_csv('./data/address_matching_data.csv')
train_df.loc[train_df['is_match']==-1, 'is_match'] = 0
test_df = pd.read_csv('./data/address_matching_test.csv')
test_df['is_match'] = np.nan


# In[3]:


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[4]:


df_ = train_df.append(test_df)
use_columns = [s for s in df_.columns if s not in ['id', 'is_match']]
df = df_[use_columns]

# handle NA
for c in df.columns:
    if df[c].dtype=='object':
        df.loc[df[c]=='?', c]=0
    else:
        print('skip ', c)


# In[5]:


# convert to matrix
categories = ['phone_equality', 'fax_equality', 'street_number_equality']
for c in df.columns:
    if c in categories: continue
    df[c] = df[c].astype(np.float32)
    
df, cols = one_hot_encoder(df, nan_as_category=False)


# In[6]:


df['is_match'] = df_['is_match']


# In[13]:


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, num_folds, stratified = False, debug= False):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    
    regs = []
    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['is_match'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['is_match'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['is_match'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                   'max_depth':-1,
                   'n_estimators':300,
                   'learning_rate':0.05,
                   'num_leaves':2**12-1,
                   'colsample_bytree':0.28,
                   'objective':'binary', 
                   'n_jobs':-1
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )
        regs.append(reg)

    return regs


# In[14]:


# separate train set and test set
train_df = df[df['is_match'].notnull()]
test_df = df[df['is_match'].isnull()]

FEATS_EXCLUDED = ['is_match']


# In[16]:


# start training on train_df
regs = kfold_lightgbm(train_df, num_folds=5, stratified=False, debug=False)


# In[17]:


# start testing on test_df
sub_preds = np.zeros(test_df.shape[0])
feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
for reg in regs:
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / len(regs)


# In[18]:


preds = sub_preds.copy()
preds[preds<0.5]=-1
preds[preds>0.5]=1


# In[19]:


test_df = pd.read_csv('./data/address_matching_test.csv')
test_df['is_match'] = preds.astype(int)
test_df = test_df.reset_index()
test_df[['id', 'is_match']].to_csv('result.csv', index=False)

