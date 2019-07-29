# -*- coding: utf-8 -*-
%reload_ext autoreload
%autoreload 2

import os
from collections import OrderedDict

import pandas as pd

########################################
import da
import lightgbm as lgb
import tensorflow as tf
import util_date
from lightgbm.sklearn import LGBMClassifier, LGBMClassifiers
from mlens.ensemble import BlendEnsemble, SuperLearner
####Ensemble Learning
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from util_date import *
from util_feature import *
from util_model import *
from util_plot import *
from util_text import *

dir0 = os.getcwd()
os.chdir( dir0 + "/da/")
print( os.getcwd() )

print(tf, tf.__version__)





EvolutionaryAlgorithmSearchCV


da.util_feature.pd_stat_histogram(df)



class tt(object) :
  def ff(self,) :
      pass


a = tt()    


a.ff()


a.ff()



folder = os.getcwd() + "/"

df = pd.read_csv(folder + '/data/address_matching_data.csv')
df.head(5)



df.sample(frac=0.3).to_csv(folder + '/data/address_small.csv', index=False  )




df = df.replace( "?", np.nan )



colname=[ "city_levenshtein_simple" ]
colexclude=None
bins=5
suffix="_bin"
method=""

colexclude = [] if colexclude is None else  colexclude
colname = colname if colname is not None else list(df.columns)
colnew  = []
col_stat = OrderedDict()
for c in colname:
        print(c,)
        if c in colexclude:
            continue

        df[c] = df[c].astype(np.float32)
        mi, ma = df[c].min(), df[c].max()
        space = (ma - mi) / bins
        lbins = [mi + i * space for i in range(bins+1)]
        lbins[0] -= 0.0001

        cbin = c + suffix 
        labels = np.arange(0, len(lbins)-1)
        colnew.append( cbin)
        df[cbin ] = pd.cut(df[c], bins=bins, labels=labels)
        
        #### NA processing
        df[cbin ] = df[cbin ].astype("int") 
        df[cbin ] = df[cbin ].apply( lambda x : x if  x >= 0.0 else -1)  ##3 NA Values
        col_stat[cbin] = df.groupby( cbin ).agg({  c : {"size", "min", "mean", "max"}  })
        print(col_stat[cbin] )


df[c + suffix ] = df[c + suffix ].fillna(-1) 



colid = "id"
colnum = ['name_levenshtein_simple', 'name_trigram_simple',
        'name_levenshtein_term', 'name_trigram_term', 'city_levenshtein_simple',
        'city_trigram_simple', 'city_levenshtein_term', 'city_trigram_term',
        'zip_levenshtein_simple', 'zip_trigram_simple', 'zip_levenshtein_term',
        'zip_trigram_term', 'street_levenshtein_simple',
        'street_trigram_simple', 'street_levenshtein_term',
        'street_trigram_term', 'website_levenshtein_simple',
        'website_trigram_simple', 'website_levenshtein_term',
        'website_trigram_term', 'phone_levenshtein', 'phone_trigram',
        'fax_levenshtein', 'fax_trigram', 'street_number_levenshtein',
        'street_number_trigram']

colcat = ['street_number_trigram', 'phone_equality', 'fax_equality',
          'street_number_equality']
coltext =[]

coltarget = "is_match"



df, colbin_stat = pd_colnum_tocat(df, colname=colnum, colexclude=None, 
                                  bins=5, suffix="_bin", method="")



dfnum_hot = pd_col_to_onehot(df[colnum_bin], colname, returncol=0)





datestring_to












clf = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l2', 
                        n_estimators = 20, num_leaves = 38)


param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)



        


param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
gridsearch = GridSearchCV(clf, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)




https://github.com/flennerhag/mlens





ensemble = SuperLearner()
ensemble.add(estimators)
ensemble.add_meta(meta_estimator)
ensemble.fit(X, y).predict(X)


# Build the first layer
ensemble.add( clf_log)
ensemble.add( clf_log)





# Attach the final meta estimator
ensemble.add_meta(LogisticRegression())

# --- Use ---

# Fit ensemble
ensemble.fit(X[:75], y[:75])

# Predict
preds = ensemble.predict(X[75:])






# ensemble = SuperLearner(scorer=roc_auc_score, random_state=32, verbose=2)


def model_ensemble_build(clf_list, proba, **kwargs):
    """Return an ensemble."""
    ensemble = BlendEnsemble(**kwargs)
    ensemble.add(clf_list, proba=proba)   # Specify 'proba' here
    ensemble.add_meta(LogisticRegression())
    return ensemble


clf_list =  [ clf_log, clf_lgb]

clf_ens = model_ensemble_build(clf_list, proba=True, scorer=roc_auc_score )


print(clf_ens)








        

##########################################################################
df = pd.read_csv( "data/titanic_train.csv")


df1 = pd_colnum_tocat_quantile( df, colname=[ "Fare" ],   bins=5,
                          suffix="_bin" )

df[ "Fare_bin" ] 




def pd_col_findtype(df) :
  """
  :param df:
  :return:
  """
  n = len(df) + 0.0
  colcat , colnum, coldate, colother = [], [], [], []
  for x in df.columns :
      nunique = len( df[x].unique())
      ntype = str(df[x].dtype)
      r =  nunique /n
      print(r, nunique, ntype )


      if r > 0.90 :
          colother.append(x)


      elif nunique < 3 :
          colcat.append(x)

      elif ntype == "o"  :
          colcat.append(x)

      elif nunique > 50 and ( "float" in ntype or  "int" in ntype ):
          colnum.append(x)

      else :
          colother.append(x)

  return colcat , colnum, coldate, colother



pd_col_findtype(df) 













"""
df[ "Fare_bin" ] = df[ "Fare_bin" ].astype("int")
df[[ "Fare_bin" ]].hist()

df[[ "Fare_bin", "Fare" ]]









c = "Fare"


"""
