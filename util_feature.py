# -*- coding: utf-8 -*-
import os, sys
import numpy as np, gc, pandas as pd, copy
#import sqlalchemy as sql, dask.dataframe as dd, dask
from attrdict import AttrDict as dict2
import arrow
from time import time
import copy
import gc
gc.collect()


#####################################################################################
#####################################################################################
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import copy

import sklearn as sk
from sklearn import manifold, datasets
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import random
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


try :
  from catboost import CatBoostClassifier, Pool, cv
except Exception as e:
    print(e)




print('os.getcwd', os.getcwd())




#####################################################################################
def pd_tohot(df, colnames) :
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




def pd_num_tocat(df ):
  for c in df.columns:
    if c in categories: continue
    df[c] = df[c].astype(np.float32)
    mi,ma = df[c].min(), df[c].max()
    space=(ma-mi)/5
    bins = [mi+i*space for  i in range(6)]
    bins[0]-=0.0000001
    df[c] = pd.cut(df[c], bins=bins, labels=labels)

        





#####################################################################################
#####################################################################################
def feature_impt_logis(clf, cols2) :
    
  dfeatures = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
  dfeatures['rank'] = np.arange(0, len(dfeatures))
  return dfeatures






#####################################################################################
#####################################################################################
def merge_columns( dfm3, ll0 ) :
  dd = {}
  for x in ll0 :
     ll2 = [] 
     for t in dfm3.columns :
       if x in t and t[len(x):len(x)+1] == "_" :
           ll2.append(t)

     dd[x]= ll2
  return dd

def merge_colunns2( dfm3,  l, x0 ) :
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










#####################################################################################
#####################################################################################
def pd_downsample(df, coltarget="y", n1max= 10000, n2max= -1, isconcat=1 ) :    
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
    
   else :
     print("y=1", n1, "y=0", n0)
     return df0, df1




def ccol(df):
    return list(df.columns)




def plotxy(x,y, color=1, size=1, title= "") :
  color = np.zeros(len(x)) if type(color) == int else color  
  fig, ax = plt.subplots(figsize=(12, 10))
  plt.scatter( x , y,  c= color, cmap="Spectral", s=size)
  plt.title(   title, fontsize=11 )
  plt.show()

    


def get_dfna_col(dfm2) :
 ll = []
 for x in dfm2.columns :
   nn = dfm2[ x ].isnull().sum()     
   nn = nn + len(dfm2[ dfm2[x] == -1 ])
   
   ll.append(nn)
 dfna_col = pd.DataFrame( {'col': list(dfm2.columns), 'n_na': ll} )
 dfna_col['n_tot'] = len(dfm2)
 dfna_col['pct_na'] = dfna_col['n_na'] / dfna_col['n_tot']
 return dfna_col


def get_dfna_user(dfm2, n = 10**6) :
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





def get_stat_imbalance(df):
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
    



def cat_correl_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



def cat_correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta



def theils_u(x, y):
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
def get_kaiso_limit(dfm2, col_score='scoress', coldefault="y", ntotal_default=491, def_list=None, nblock=20.0) : 
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
def get_kaiso2(x, l1) :
   for i in range(0, len(l1)) :
       if x >= l1[i]  :
           return i+1
   return i + 1+1




#### Histo     
def np_histo(dfm2, bins=50, col0='diff', col1='y') :
  hh  = np.histogram( dfm2[ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2 = pd.DataFrame({ 'xall' : hh[1][:-1] , 
                       'freqall' :hh[0] } )[[ 'xall', 'freqall' ]]
  hh2['densityall'] = hh2['freqall'] / hh2['freqall'].sum()    

        
  hh  = np.histogram( dfm2[ dfm2[ col1 ] == 0 ][ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2['x0'] = hh[1][:-1]
  hh2['freq0'] = hh[0]
  hh2['density0'] = hh2['freq0'] / hh2['freq0'].sum()

  
  hh  = np.histogram( dfm2[ dfm2[ col1 ] == 1 ][ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2['x1'] = hh[1][:-1]
  hh2['freq1'] = hh[0]
  hh2['density1'] = hh2['freq1'] / hh2['freq1'].sum()
  
  return hh2  



##### Drop duplicates
from collections import OrderedDict
def np_drop_duplicates(l1):
  l0 = list( OrderedDict((x, True) for x in l1 ).keys())
  return l0


def col_extract_colbin( cols2) :
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



def col_stats(df) :
  ll = { 'col' : [], 'nunique' : [] }
  for x in df.columns:
     ll['col'].append( x )
     ll['nunique'].append( df[x].nunique() )
  ll =pd.DataFrame(ll)
  n =len(df) + 0.0
  ll['ratio'] =  ll['nunique'] / n
  ll['coltype'] = ll['nunique'].apply( lambda x :  'cat' if x < 100 else 'num')
  
  return ll



def np_intersection(df1, df2, colid) :
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



def pd_feat_check( dfm2 ) :
 # print(dfm2['segmentc'].values[0]) 
 for x in dfm2.columns :
    if len( dfm2[x].unique() ) > 2 and dfm2[x].dtype  != np.dtype('O'):
        print(x, len(dfm2[x].unique())  ,  dfm2[x].min() , dfm2[x].max()  )



def split_train(df1, ntrain=10000, ntest=100000, colused=None ) :
  n1  = len( df1[ df1['y'] == 0 ] )
  dft = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntest, False), : ]  , 
                    df1[ (df1['y'] == 1) & (df1['def'] > 201803 ) ].iloc[ : , :]  ))

  X_test = dft[ colused ].values 
  y_test = dft[ 'y'  ].values
  print('test', sum(y_test))

  ######## Train data   
  n1   = len( df1[ df1['y'] == 0 ] )
  dft2 = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntrain, False), : ]  , 
                      df1[ ( df1['y'] == 1) & (df1['def'] > 201703 ) & (df1['def'] < 201804 )  ].iloc[ : , :]  ))
  dft2 = dft2.iloc[ np.random.choice( len(dft2) , len(dft2), False) , : ]

  X_train = dft2[ colused ].values 
  y_train = dft2[ 'y' ].values
  print('train', sum(y_train))
  return X_train, X_test, y_train, y_test 



def split_train2(df1, ntrain=10000, ntest=100000, colused=None, nratio =0.4 ) :
  n1  =  len( df1[ df1['y'] == 0 ] )
  n2  =  len( df1[ df1['y'] == 1 ] ) 
  n2s =  int(n2*nratio)  # 80% of default
      
  #### Test data
  dft = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntest, False), : ]  , 
                    df1[ (df1['y'] == 1)  ].iloc[: , :]  ))

  X_test = dft[ colused ].values 
  y_test = dft[ 'y'  ].values
  print('test', sum(y_test))

  ######## Train data   
  n1   = len( df1[ df1['y'] == 0 ] )
  dft2 = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntrain, False), : ]  , 
                     df1[ (df1['y'] == 1)  ].iloc[ np.random.choice( n2 , n2s, False) , :]   ))
  dft2 = dft2.iloc[ np.random.choice( len(dft2) , len(dft2), False) , : ]

  X_train = dft2[ colused ].values 
  y_train = dft2[ 'y' ].values
  print('train', sum(y_train))
  return X_train, X_test, y_train, y_test 








#####################################################################################
def pd_remove(df, cols) :
 for x in  cols :
   try :   
    del df[ x ]
   except : pass
 return df

def sk_showconfusion( Y_train,Y_pred, isprint=True):
  cm = sk.metrics.confusion_matrix(Y_train, Y_pred); cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  if isprint: print(( cm_norm[0,0] + cm_norm[1,1])); print(cm_norm); print(cm)
  return cm, cm_norm, cm_norm[0,0] + cm_norm[1,1]


def sk_feature_importance(clf, cols) :
  dfcol = pd.DataFrame( {  'col' :  cols,  'feat':   clf.feature_importances_
                     } ).sort_values('feat', ascending=0).reset_index()
  dfcol['cum'] = dfcol['feat'].cumsum(axis = 0) 
  colsel = list( dfcol[ dfcol['cum'] < 0.9 ]['col'].values )
  return dfcol, colsel


def sk_showmetrics(y_test, ytest_pred, ytest_proba,target_names=['0', '1']) :
  #### Confusion matrix
  mtest  = sk_showconfusion( y_test, ytest_pred, isprint=False)
  # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
  auc  =  roc_auc_score( y_test, ytest_proba)   # 
  gini =  2*auc -1
  acc  =  accuracy_score(  y_test, ytest_pred )
  f1macro = sk.metrics.f1_score(y_test, ytest_pred, average='macro')
  
  print('Test confusion matrix')
  print(mtest[0]) ; print(mtest[1])
  print('auc ' + str(auc) )
  print('gini '+str(gini) )
  print('acc ' + str(acc) )
  print('f1macro ' + str(f1macro) )
  print('Nsample ' + str(len(y_test)) )
  
  print(classification_report(y_test, ytest_pred, target_names=target_names))

  # calculate roc curve
  fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curve')
  plt.show()



def sk_showmetrics2(y_test, ytest_pred, ytest_proba,target_names=['0', '1']) :
  #### Confusion matrix
  # mtest  = sk_showconfusion( y_test, ytest_pred, isprint=False)
  # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
  auc  =  roc_auc_score( y_test, ytest_proba)   # 
  gini =  2*auc -1
  acc  =  accuracy_score(  y_test, ytest_pred )
  return auc, gini, acc



def clf_prediction_score(clf, df1 , cols, outype='score') :
  def score_calc(yproba , pnorm = 1000.0 ) :
    yy =  np.log( 0.00001 + (1 - yproba )  / (yproba + 0.001) )   
    # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
    # return  np.maximum( 0.01 , yy )    ## Error it bias proba
    return yy


  X_all = df1[ cols ].values 

  yall_proba   = clf.predict_proba(X_all)[:, 1]
  yall_pred    = clf.predict(X_all)
  try :
    y_all = df1[ 'y'  ].values
    sk_showmetrics(y_all, yall_pred, yall_proba)
  except : pass

  yall_score   = score_calc( yall_proba )
  yall_score   = 1000 * ( yall_score - np.min( yall_score ) ) / (  np.max(yall_score) - np.min(yall_score) )

  if outype == 'score' :
      return yall_score
  if  outype == 'proba' :
      return yall_proba, yall_pred




def col_extract(colbin) :
 '''
    Column extraction 
 '''   
 colnew = []
 for x in colbin :
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



def col_remove(cols, colsremove) :
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




def col_remove_fuzzy(cols, colsremove) :
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





def pd_filter(dfxx , cols ) :      
  df1  = copy.deepcopy( dfxx[cols + [ 'def' , 'y'] ] )
  df1  = df1[  df1['def'] < 201905  ] 
  df1  = df1[ (df1['def'] > 201703) | (df1['def'] == -1 )  ] 
  return  df1











