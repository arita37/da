"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas

"""
from collections import OrderedDict
# -*- coding: utf-8 -*-
import os, sys
import numpy as np, gc, pandas as pd, copy

import dask.dataframe as dd, dask
from attrdict import AttrDict as dict2
import arrow
from time import time
import copy
import gc

import sklearn as sk
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import random



import copy
import itertools
import math
import os
import re
import sys
from calendar import isleap
from collections import OrderedDict
from datetime import datetime, timedelta

import arrow
import numpy as np
import pandas as pd
import requests
import scipy as sci
import sklearn as sk

from dateutil.parser import parse
from sklearn import covariance, linear_model, model_selection

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import util
from attrdict import AttrDict as dict2
from tabulate import tabulate


try :
  from catboost import CatBoostClassifier, Pool, cv
except Exception as e:
    print(e)




print('os.getcwd', os.getcwd())




def pd_tohot(df, colnames):
    
     for x in colnames :
       print( x, df.shape , flush=True)
       try :   
        nunique = len( df[x].unique() )
         
        if nunique > 2  :  
          df = pd.concat([df , pd.get_dummies(df[x], prefix= x)],axis=1).drop( [x],axis=1)
          # coli =   [ x +'_' + str(t) for t in  lb.classes_ ] 
          # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
          # del df[x]
        else :
          lb = preprocessing.LabelBinarizer()  
          vv = lb.fit_transform(df[x])  
          df[x] = vv
       except : pass
     return df


def pd_one_hot_encoder(df, nan_as_category = True, categorical_columns=None):
    
    original_columns = list(df.columns)
#     categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    cat_col = [col for col in df.columns]
    df = pd.get_dummies(df, columns= cat_col, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



def pd_num_tocat(df):
    
    for c in df.columns:
        if c in categories: continue
        df[c] = df[c].astype(np.float32)
        mi,ma = df[c].min(), df[c].max()
        space=(ma-mi)/5
        bins = [mi+i*space for  i in range(6)]
        bins[0]-=0.0000001
        df[c] = pd.cut(df[c], bins=bins, labels=labels)

   
def sk_feature_impt_logis(clf, cols2) :
    
    dfeatures = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
    dfeatures['rank'] = np.arange(0, len(dfeatures))
    return dfeatures

def pd_merge_columns( dfm3, ll0 ) :
    
    dd = {}
    for x in ll0 :
       ll2 = [] 
       for t in dfm3.columns :
         if x in t and t[len(x):len(x)+1] == "_" :
             ll2.append(t)
       dd[x]= ll2
    return dd


def pd_merge_colunns2( dfm3,  l, x0 ) :
    
    dfz = pd.DataFrame( { 'easy_id' : dfm3['easy_id'].values })  
    for t in l :
       ix  =  t.rfind("_") 
       val = int( t[ix+1:])
       print(ix, t[ix+1:] )
       dfz[t] = dfm3[t].apply( lambda x : val if x > 0 else 0 )

  # print(dfz)
    dfz = dfz.set_index('easy_id')
    dfz[x0] = dfz.iloc[:,:].sum(1)
    for t in dfz.columns :
       if t != x0 :
           del dfz[t]
    return dfz

    
   


def pd_downsample(df, coltarget="y", n1max= 10000, n2max= -1, isconcat=1 ):
    """
    DownSampler      
    """ 
    #n0  = len( df1 )
    #l1  = np.random.choice( len(df1) , n1max, replace=False)

    df1 = df[ df[coltarget] == 0 ].sample(n= n1max)


    #df1   = df[ df[coltarget] == 1 ] 
    #n1    = len(df1 )
    #print(n1)
    n2max = len(  df[ df[coltarget] == 1 ]  ) if n2max == -1 else n2max
    #l1    = np.random.choice(len(df1) , n2max, replace=False)
    df0   = df[ df[coltarget] == 1 ].sample(n= n2max)
    #print(len(df0))

    if isconcat :
        df2 = pd.concat(( df1, df0 ))   
        df2 = df2.sample(frac=1.0, replace=True)
        return df2
    
    else:
        print("y=1", n1, "y=0", n0)
        return df0, df1





def pd_stat_na_percol(dfm2):
    
    ll = []
    for x in dfm2.columns :
      nn = dfm2[ x ].isnull().sum()     
      nn = nn + len(dfm2[ dfm2[x] == -1 ])
      
      ll.append(nn)
    dfna_col = pd.DataFrame( {'col': list(dfm2.columns), 'n_na': ll} )
    dfna_col['n_tot'] = len(dfm2)
    dfna_col['pct_na'] = dfna_col['n_na'] / dfna_col['n_tot']
    return dfna_col



def pd_stat_na_perow(dfm2, n = 10**6):
    
    ll =[]  ; n = 10**6
    for ii,x in dfm2.iloc[ :n, :].iterrows() :
       ii = 0
       for t in x:
         if pd.isna(t) or t == -1 :
           ii = ii +1
       ll.append(ii)
    dfna_user = pd.DataFrame( {''    : dfm2.index.values[:n] , 'n_na': ll,
                              'n_ok' : len(dfm2.columns) - np.array(ll) } )
    return dfna_user



def pd_stat_col_imbalance(df):
    
    ll =  { x : []  for x in   [ 'col', 'xmin_freq', 'nunique', 'xmax_freq' ,'xmax' , 
                                  'xmin',  'n', 'n_na', 'n_notna'  ]   }
      
    nn = len(df)
    for x in df.columns :
       try : 
        xmin = df[x].min()
        ll['xmin_freq'].append(  len(df[ df[x] < xmin + 0.01 ]) )
        ll['xmin'].append( xmin )
        
        xmax = df[x].max()
        ll['xmax_freq'].append( len(df[ df[x] > xmax - 0.01 ]) )
        ll['xmax'].append( xmax )
        
        n_notna = df[x].count()  
        ll['n_notna'].append(  n_notna   ) 
        ll['n_na'].append(   nn - n_notna  ) 
        ll['n'].append(   nn   ) 
         
        ll['nunique'].append(   df[x].nunique()   ) 
        ll['col'].append(x)
       except : pass
    
    
    ll = pd.DataFrame(ll)
    ll['xmin_ratio'] = ll['xmin_freq'] / nn
    ll['xmax_ratio'] = ll['xmax_freq'] / nn
    return ll




def np_cat_correlation_ratio(x,y):
    
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def np_mutual_info_(x, y):
        
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
  
#### Calculate KAISO Limit  #########################################################
def pd_segment_limit(dfm2, col_score='scoress', coldefault="y", ntotal_default=491, def_list=None, nblock=20.0) : 
    
    if def_list is None :
       def_list = np.ones(21) * ntotal_default / nblock       
  
    dfm2['scoress_bin'] = dfm2[ col_score ].apply(lambda x : np.floor( x / 1.0 ) * 1.0  )
    dfs5 = dfm2.groupby(  'scoress_bin' ).agg( { col_score :  'mean'  ,
                                               coldefault :  {'sum', 'count'}
    }).reset_index()
    dfs5.columns = [ x[0]  if x[0] == x[1] else  x[0] +'_'+ x[1]  for x in  dfs5.columns ] 
    dfs5 = dfs5.sort_values( col_score , ascending=False) 
  # return dfs5

    l2 =  []
    k  =  1
    ndef, nuser = 0 , 0
    for i, x in dfs5.iterrows() :
      if k > nblock : break  
      nuser =  nuser + x[ coldefault + '_count']    
      ndef  =  ndef  + x[ coldefault + '_sum']  
      pdi   =  ndef / nuser 
  
      if ndef > def_list[k-1] :
    #if  pdi > pdlist[k] :
        l2.append( [np.round( x[ col_score ], 1) ,  k, pdi,  ndef, nuser ] )      
        k = k + 1
        ndef, nuser = 0 , 0
    l2.append( [np.round( x[ col_score ], 1 ) ,  k, pdi,  ndef, nuser ] )  
    l2  = pd.DataFrame(l2, columns=[  col_score , 'kaiso3' , 'pd', 'ndef', 'nuser' ] )
    return l2


##### Get Kaiso limit ###############################################################
def get_kaiso2(x, l1):
    
    for i in range(0, len(l1)) :
        if x >= l1[i]  :
            return i+1
    return i + 1+1


def np_drop_duplicates(l1):    
    l0 = list( OrderedDict((x, True) for x in l1 ).keys())
    return l0


def np_col_extractname_colbin(cols2) :
    coln = []
    for ss in cols2 :
     xr = ss[ss.rfind("_")+1:]
     xl = ss[:ss.rfind("_")]
     if len(xr) < 3 :  # -1 or 1
       coln.append( xl )
     else :
       coln.append( ss )     
        
    coln = np_drop_duplicates(coln)
    return coln


def pd_stats_col(df) :
    ll = { 'col' : [], 'nunique' : [] }
    for x in df.columns:
       ll['col'].append( x )
       ll['nunique'].append( df[x].nunique() )
    ll =pd.DataFrame(ll)
    n =len(df) + 0.0
    ll['ratio'] =  ll['nunique'] / n
    ll['coltype'] = ll['nunique'].apply( lambda x :  'cat' if x < 100 else 'num')
  
    return ll


def pd_col_intersection(df1, df2, colid) :
    n2 = list( set(df1[colid].values).intersection(df2[colid]) )
    print("total matchin",  len(n2), len(df1), len(df2) )
    return n2


def pd_feat_normalize(dfm2, colnum_log, colproba) :
    
    for x in [  'SP1b' , 'SP2b'  ] :
      dfm2[x] = dfm2[x] * 0.01

    dfm2['SP1b' ] = dfm2['SP1b' ].fillna(0.5)  
    dfm2['SP2b' ] = dfm2['SP2b' ].fillna(0.5)

    for x in colnum_log :
      try :  
        dfm2[x] =np.log( dfm2[x].values.astype(np.float64)  + 1.1 )
        dfm2[x] = dfm2[x].replace(-np.inf, 0)
        dfm2[x] = dfm2[x].fillna(0)
        print(x, dfm2[x].min(), dfm2[x].max() )
        dfm2[x] = dfm2[x] / dfm2[x].max()
      except :
        pass
          
  
    for x in colproba :
      print(x)  
      dfm2[x] = dfm2[x].replace(-1, 0.5)
      dfm2[x] = dfm2[x].fillna(0.5)
    
    return dfm2




def pd_col_check( dfm2 ) :
    for x in dfm2.columns :
       if len( dfm2[x].unique() ) > 2 and dfm2[x].dtype  != np.dtype('O'):
           print(x, len(dfm2[x].unique())  ,  dfm2[x].min() , dfm2[x].max()  )


def pd_col_remove(df, cols) :
    
    for x in  cols :
      try :   
       del df[ x ]
      except : pass
    return df



def sk_feature_importance(clfrf, feature_name):
    importances = clfrf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(0, len(feature_name)):
        if importances[indices[f]] > 0.0001:
            print(
                str(f + 1), str(indices[f]), feature_name[indices[f]], str(importances[indices[f]])
            )


def np_col_extractname(col_onehot):
    
    '''
    Column extraction 
    '''   
    colnew = []
    for x in col_onehot :
       if len(x) > 2 :
         if   x[-2] ==  "_" :
             if x[:-2] not in colnew : 
               colnew.append(  x[:-2] ) 
              
         elif x[-2] ==  "-"   :
             if  x[:-3] not in colnew :
               colnew.append(  x[:-3]  )
              
         else :
             if x not in colnew :
                colnew.append( x ) 
    return colnew




def np_col_remove(cols, colsremove) :
    
    #cols = list(df1.columns)
    '''
    colsremove = [
    'y', 'def',
    'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
    'score' ,   'segment2' ,
    'scoreb', 'score_kaisob', 'segmentb', 'def_test'
    ]
    colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    '''
    for x in colsremove :
      try :     cols.remove(x)
      except :  pass
    return cols



def np_col_remove_fuzzy(cols, colsremove) :
    
    #cols = list(df1.columns)
    '''
      colsremove = [
         'y', 'def',
         'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
         'score' ,   'segment2' ,
         'scoreb', 'score_kaisob', 'segmentb', 'def_test'
      ]
      colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    '''
    cols3 = []
    for t in cols :
      flag = 0   
      for x in colsremove :
          if x in t :
              flag = 1
              break
      if flag == 0 :
        cols3.append(t)
    return cols3


def pd_feature_filter(dfxx , cols ) :      
    df1  = copy.deepcopy( dfxx[cols + [ 'def' , 'y'] ] )
    df1  = df1[  df1['def'] < 201905  ] 
    df1  = df1[ (df1['def'] > 201703) | (df1['def'] == -1 )  ] 
    return  df1



def col_feature_importance(Xcol, Ytarget):
    """ random forest for column importance """
    pass

def col_study_getcategorydict_freq(catedict):
    
    """ Generate Frequency of category : Id, Freq, Freqin%, CumSum%, ZScore
      given a dictionnary of category parsed previously
  """
    catlist = []
    for key, v in list(catedict.items()):
        df = util.pd_array_todataframe(util.np_dict_tolist(v), ["category", "freq"])
        df["freq_pct"] = 100.0 * df["freq"] / df["freq"].sum()
        df["freq_zscore"] = df["freq"] / df["freq"].std()
        df = df.sort_values(by=["freq"], ascending=0)
        df["freq_cumpct"] = 100.0 * df["freq_pct"].cumsum() / df["freq_pct"].sum()
        df["rank"] = np.arange(0, len(df.index.values))
        catlist.append((key, df))
    return catlist
  
  
      
def col_pair_correl(Xcol, Ytarget):
    pass


def col_pair_interaction(Xcol, Ytarget):
    """ random forest for pairwise interaction """
    pass


def col_study_summary(Xmat=[0.0, 0.0], Xcolname=["col1", "col2"], Xcolselect=[9, 9], isprint=0):
    
    n, m = np.shape(Xmat)
    if Xcolselect == [9, 9]:
        Xcolselect = np.arange(0, m)
    if len(Xcolname) != m:
        print("Error column size: ")
        return None
    colanalysis = []
    for icol in Xcolselect:
        Xraw_1unique = np.unique(Xmat[:, icol])
        vv = [
            Xcolname[icol],
            icol,
            len(Xraw_1unique),
            np.min(Xraw_1unique),
            np.max(Xraw_1unique),
            np.median(Xraw_1unique),
            np.mean(Xraw_1unique),
            np.std(Xraw_1unique),
        ]
        colanalysis.append(vv)

    colanalysis = pd.DataFrame(
        colanalysis,
        columns=[
            "Col_name",
            "Col_id",
            "Nb_Unique",
            "MinVal",
            "MaxVal",
            "MedianVal",
            "MeanVal",
            "StdDev",
        ],
    )
    if isprint:
        print(("Nb_Samples:", np.shape(Xmat)[0], "Nb Col:", len(Xcolname)))
        print(colanalysis)
    return colanalysis



def pd_filter_column(df_client_product, filter_val=[], iscol=1):
    """
   # Remove Columns where Index Value is not in the filter_value
   # filter1= X_client['client_id'].values
   :param df_client_product:
   :param filter_val:
   :param iscol:
   :return:
   """
    axis = 1 if iscol == 1 else 0
    col_delete1 = []
    for colname in df_client_product.index.values:  # !!!! row Delete
        if colname in filter_val:
            col_delete1.append(colname)

    df2 = df_client_product.drop(col_delete1, axis=axis, inplace=False)
    return df2

def pd_missing_show():
    

    """
   https://blog.modeanalytics.com/python-data-visualization-libraries/


   Missing Data

     missingno
import missingno as msno
%matplotlib inline
msno.matrix(collisions.sample(250))
At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be completely populated, while geographic information seems mostly complete, but spottier.

The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum rows.

This visualization will comfortably accommodate up to 50 labelled variables. Past that range labels begin to overlap or become unreadable, and by default large displays omit them.


Heatmap
The missingno correlation heatmap lets you measure how strongly the presence of one variable positively or negatively affect the presence of another:
msno.heatmap(collisions)


https://github.com/ResidentMario/missingno

   """


def pd_describe(df):
    """ Describe the tables


   """
    coldes = [
        "col",
        "coltype",
        "dtype",
        "count",
        "min",
        "max",
        "nb_na",
        "pct_na",
        "median",
        "mean",
        "std",
        "25%",
        "75%",
        "outlier",
    ]

    def getstat(col, type1="num"):
        """
         max, min, nb, nb_na, pct_na, median, qt_25, qt_75,
         nb, nb_unique, nb_na, freq_1st, freq_2th, freq_3th
         s.describe()
         count    3.0  mean     2.0 std      1.0
         min      1.0   25%      1.5  50%      2.0
         75%      2.5  max      3.0
      """
        ss = list(df[col].describe().values)
        ss = [str(df[col].dtype)] + ss
        nb_na = df[col].isnull().sum()
        ntot = len(df)
        ss = ss + [nb_na, nb_na / (ntot + 0.0)]

        return pd.Series(
            ss,
            ["dtype", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "nb_na", "pct_na"],
        )

    dfdes = pd.DataFrame([], columns=coldes)
    cols = df.columns
    for col in cols:
        dtype1 = str(df[col].dtype)
        if dtype1[0:3] in ["int", "flo"]:
            row1 = getstat(col, "num")
            dfdes = pd.concat((dfdes, row1))

        if dtype1 == "object":
            pass


def pd_stack_dflist(df_list):
    df0 = None
    for i, dfi in enumerate(df_list):
        if df0 is None:
            df0 = dfi
        else:
            try:
                df0 = df0.append(dfi, ignore_index=True)
            except:
                print(("Error appending: " + str(i)))
    return df0


def pd_validation_struct():
    pass
    """
  https://github.com/jnmclarty/validada

  https://github.com/ResidentMario/checkpoints


  """


def pd_checkpoint():
    pass


"""
  Create Checkpoint on dataframe to save intermediate results
  https://github.com/ResidentMario/checkpoints
  To start, import checkpoints and enable it:

from checkpoints import checkpoints
checkpoints.enable()
This will augment your environment with pandas.Series.safe_map and pandas.DataFrame.safe_apply methods. Now suppose we create a Series of floats, except for one invalid entry smack in the middle:

import pandas as pd; import numpy as np
rand = pd.Series(np.random.random(100))
rand[50] = "____"
Suppose we want to remean this data. If we apply a naive map:

rand.map(lambda v: v - 0.5)

    TypeError: unsupported operand type(s) for -: 'str' and 'float'
Not only are the results up to that point lost, but we're also not actually told where the failure occurs! Using safe_map instead:

rand.safe_map(lambda v: v - 0.5)

    <ROOT>/checkpoint/checkpoints/checkpoints.py:96: UserWarning: Failure on index 50
    TypeError: unsupported operand type(s) for -: 'str' and 'float'


"""


"""
You can control how many decimal points of precision to display
In [11]:
pd.set_option('precision',2)

pd.set_option('float_format', '{:.2f}'.format)


Qtopian has a useful plugin called qgrid - https://github.com/quantopian/qgrid
Import it and install it.
In [19]:
import qgrid
qgrid.nbinstall()
Showing the data is straighforward.
In [22]:
qgrid.show_grid(SALES, remote_js=True)


SALES.groupby('name')['quantity'].sum().plot(kind="bar")


"""


######################  Transformation   ###########################################################
def tf_transform_catlabel_toint(Xmat):
    """
     # ["paris", "paris", "tokyo", "amsterdam"]  --> 2 ,5,6
     # np.array(le.inverse_transform([2, 2, 1]))
     le = preprocessing.LabelEncoder()
     le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
list(le.classes_)
['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
   """
    le = sk.preprocessing.LabelEncoder()
    ncol = Xmat.shape[1]
    Xnew = np.zeros_like(Xmat)
    mapping_cat_int = {}

    for k in range(0, ncol):
        Xnew[:, k] = le.fit_transform(Xmat[:, k])
        mapping_cat_int[k] = le.get_params()

    return Xnew, mapping_cat_int


def tf_transform_pca(Xmat, dimpca=2, whiten=True):
    """Project ndim data into dimpca sub-space  """
    pca = PCA(n_components=dimpca, whiten=whiten).fit(Xmat)
    return pca.transform(Xmat)



######################### OPTIM   ###################################################
def optim_is_pareto_efficient(Xmat_cost, epsilon=0.01, ret_boolean=1):
    """ Calculate Pareto Frontier of Multi-criteria Optimization program
    c1, c2  has to be minimized : -Sharpe, -Perf, +Drawdown
    :param Xmat_cost: An (n_points, k_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    pesp = 1.0 + epsilon  # Relax Pareto Constraints
    is_efficient = np.ones(Xmat_cost.shape[0], dtype=bool)
    for i, c in enumerate(Xmat_cost):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                Xmat_cost[is_efficient] <= c * pesp, axis=1
            )  # Remove dominated points
    if ret_boolean:
        return is_efficient
    else:
        return Xmat_cost[is_efficient]
    # return is_efficient






