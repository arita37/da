
import os, sys, gc, glob,  copy, json,zlib, random
import platform, socket, subprocess
import pandas as pd, numpy as np
import time, msgpack
from collections import defaultdict


import uuid
from six.moves import queue
#from time import sleep, time
from dateutil.relativedelta import relativedelta
import datetime
import _pickle as cPickle


########################################################################################################################
VERBOSE = 10


########################################################################################################################
# import scipy.sparse as sp
# from sklearn.datasets        import fetch_mldata
#from sklearn.preprocessing   import scale
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, accuracy_score
#from scipy.sparse import csr_matrix

###############################################################################
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import sklearn as sk
from sklearn import manifold, datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_iris


from sklearn.linear_model import LinearRegression,ElasticNet, RidgeCV, RANSACRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (confusion_matrix,roc_curve, roc_auc_score, classification_report,
                             accuracy_score)



####################################################################################################
import zlocal


def log(*s, **kw):
    print(*s, flush=True, **kw)

def log_time(*s):
    import datetime
    print(str(datetime.datetime.now()) , *s, flush=True)




####################################################################################################
def pd_cartesian(df1, df2) :
  col1 =  list(df1.columns)  
  col2 =  list(df2.columns)    
  df1['xxx'] = 1
  df2['xxx'] = 1    
  df3 = pd.merge(df1, df2,on='xxx')[ col1 + col2 ]
  return df3



def os_run_cmd_list(ll, log_filename, sleep_sec=10):
   import time, sys
   n = len(ll) 
   for ii,x in enumerate(ll):
        try :
          log(x)
          if sys.platform == 'win32' :
             cmd = f" {x}   "
          else :
             cmd = f" {x}   2>&1 | tee -a  {log_filename} "
             
          os.system(cmd)
        
          # tx= sum( [  ll[j][0] for j in range(ii,n)  ]  )
          # log(ii, n, x,  "remaining time", tx / 3600.0 )
          #log('Sleeping  ', x[0])
          time.sleep(sleep_sec)
        except Exception as e:
            log(e)


def train_split_time(df, test_period = 40, cols=None , coltime ="time_key", sort=True, minsize=5,
                     verbose=False) :  
   cols = list(df.columns) if cols is None else cols    
   if sort :
       df   = df.sort_index( ascending=1 ) 
   #imax = len(df) - test_period   
   colkey = [ t for t in cols if t not in [coltime] ]  #### All time reference be removed
   if verbose : log(colkey)
   imax = test_period
   df1  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[:max(minsize, len(dfi) -imax), :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   df2  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[max(minsize,  len(dfi) -imax):, :] ).reset_index(colkey, drop=True).reset_index(drop=True)  
   return df1, df2




def pd_histo(dfi, path_save=None, nbin=20.0, show=False) :
  q0 = dfi.quantile(0.05)
  q1 = dfi.quantile(0.95)
  dfi.hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
  if path_save is not None : plt.savefig( path_save );   
  if show : plt.show(); 
  plt.close()

    
def config_import(mod_name="offline.config.genre_l2_model", globs=None, verbose=True):
    ### from mod_name import *
    module = __import__(mod_name, fromlist=['*'])
    if hasattr(module, '__all__'):
        all_names = module.__all__
    else:
        all_names = [name for name in dir(module) if not name.startswith('_')]
    
    if verbose :
      print("Importing:", end=",")  
      for name in all_names :
         print(name, end=",")
    globs.update({name: getattr(module, name) for name in all_names})
    print("")



def os_file_check(fp):
   import os, time 
   try :
       log(fp,  os.stat(fp).st_size*0.001, time.ctime(os.path.getmtime(fp)) )  
   except :
       log(fp, "Error File Not exist")
    
class to_namespace(object):
    ## dict to namepsace
    def __init__(self, adict):
        self.__dict__.update(adict)

    def get(self, key):
        return self.__dict__.get(key)
    
def pd_add(ddict, colsname, vector, tostr=1):
   for ii, x in enumerate(colsname) :
       ddict[x].append( str(vector[ii]) if tostr else  vector[ii] )
   return ddict


def pd_filter(df, filter_dict=None ) :    
  for key,val in filter_dict.items() :
      df =   df[  (df[key] == val) ]       
  return df





def pd_read_parallel(file_list,pd_reader=None, n_pool=4, verbose=True,) :
    """
      path = root + "/data/pos/*2020011*"
      file_list = [   f for f in glob.glob( path )  ]
      dfx = pd_read_parallel(file_list,pd_reader , n_pool=4, verbose=True) 
    
    """
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=n_pool)

    def pd_reader(fp):
         return pd_read_file(path_glob=fp, verbose=False )    
        
    dfall = pd.DataFrame()
    n_file = len(file_list)
    if verbose : log_time(n_file,  n_file // n_pool )
    for j in range(n_file // n_pool +1 ) :
      log("Pool", j)  
      job_list =[]   
      for i in range(n_pool):  
         if n_pool*j + i >= n_file  : break 
         job_list.append( pool.apply_async(pd_reader, (file_list[n_pool*j + i], )))  
    
      for i in range(n_pool):  
        if i >= len(job_list): break  
        dfi   = job_list[ i].get()
        dfall = pd.concat((dfall, dfi))
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()

    
    if verbose : log_time(n_file, j * n_file//n_pool )
    return dfall



def pd_read_file2(path_glob="*.pkl", ignore_index=True,  cols=None,
                 verbose=1, nrows=-1, concat_sort=True, n_pool=1,  **kw):
  import os  
  # os.environ["MODIN_ENGINE"] = "dask"   
  # import modin.pandas as pd  
  import glob, gc,  pandas as pd, os
  readers = {
          ".pkl"     : pd.read_pickle,
          ".parquet" : pd.read_parquet,
          ".csv"     : pd.read_csv,           
          ".txt"     : pd.read_csv,
   }
  from multiprocessing.pool import ThreadPool
  pool = ThreadPool(processes=n_pool)
  
  file_list = glob.glob(path_glob)     
  # print("ok", verbose)
  dfall = pd.DataFrame()
  n_file = len(file_list)
  if verbose : log_time(n_file,  n_file // n_pool )
  for j in range(n_file // n_pool +1 ) :
      log("Pool", j, end=",")  
      job_list =[]   
      for i in range(n_pool):  
         if n_pool*j + i >= n_file  : break 
         filei         = file_list[n_pool*j + i]
         ext           = os.path.splitext(filei)[1]
         pd_reader_obj = readers[ext]                            
         job_list.append( pool.apply_async(pd_reader_obj, (filei, )))  
         if verbose : log(j, filei)
    
      for i in range(n_pool):  
        if i >= len(job_list): break  
        dfi   = job_list[ i].get()
        
        if cols is not None : dfi = dfi[cols] 
        if nrows > 0        : dfi = dfi.iloc[:nrows,:]
        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)        
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()
        
  if verbose : log_time(n_file, j * n_file//n_pool )
  return dfall  
  


def pd_read_file(path_glob="*.pkl", ignore_index=True, pd_reader=  "pd.read_pickle", cols=None,
                 verbose=1, nrows=-1, concat_sort=True,  **kw):
  import os  
  # os.environ["MODIN_ENGINE"] = "dask"   
  # import modin.pandas as pd  
  import glob, gc,  pandas as pd, os
  readers = {
          ".pkl"     : pd.read_pickle,
          ".parquet" : pd.read_parquet,
          ".csv"     : pd.read_csv,           
          ".txt"     : pd.read_csv,
   }
  
  # print("ok", verbose)
  df = pd.DataFrame()
  for ii, f in enumerate(glob.glob(path_glob)) :
      
    ext           = os.path.splitext(f)[1]
    pd_reader_obj = readers[ext]    
    dfi = pd_reader_obj(f )  ##  **kw
    
    if cols is not None :
        dfi = dfi[cols]
        
    if nrows > 0 :
        dfi = dfi.iloc[:nrows,:]

    print(ii, f) 
    if verbose > 0 : 
       print(ii, dfi.head(3), dfi.columns)  
    df = pd.concat( (df, dfi), ignore_index=ignore_index, sort= concat_sort)
    if verbose >  0: 
        print(ii, len(dfi))
    del dfi
    gc.collect()
  return df




def pd_to_onehot(df, colnames, map_dict=None, verbose=0) :
 # df = df[colnames]   
 
 for x in colnames :
   try :   
    nunique = len( df[x].unique() )
    if verbose : print( x, df.shape , nunique, flush=True)
     
    if nunique > 0  :  
      try :
         df[x] = df[x].astype("int64")
      except : pass
      # dfi = df
      
      if map_dict is not None :
        try :
          df[x] = df[x].astype( pd.CategoricalDtype( categories = map_dict[x] ) )
        except :  
          print("No map_dict for ", x)  
          
      # pd.Categorical(df[x], categories=map_dict[x] )
      prefix = x 
      dfi =  pd.get_dummies(df[x], prefix= prefix, ).astype('int32') 
      
      #if map_dict is not None :
      #  dfind =  dfi.index  
      #  dfi.index = np.arange(0, len(dfi) )
      #  dfi = dfi.T.reindex(map_dict[x]).T.fillna(0)
      #  dfi.columns = [ prefix + "_" + str(x) for x in dfi.columns ]
      #  # dfi.index = dfind 
      
      df = pd.concat([df ,dfi],axis=1).drop( [x],axis=1)
      # coli =   [ x +'_' + str(t) for t in  lb.classes_ ] 
      # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
      # del df[x]
    else :
      lb = preprocessing.LabelBinarizer()  
      vv = lb.fit_transform(df[x])  
      df[x] = vv
   except Exception as e :
     print("error", x, e, )  
 return df






###############################################################################
##### Utilities for date  #####################################################
def to_timekey(x_date):
   tstruct = x_date.timetuple()
   #tstruct = time.strptime(intdate,'%Y%m%d')
   time_key = int(time.mktime(tstruct)/86400)
   return time_key



def is_holiday(array):
    """
      is_holiday([ pd.to_datetime("2015/1/1") ] * 10)  

    """
    import holidays  
    from datetime import date
    jp_holidays = holidays.CountryHoliday('JP')
    return np.array( [ 1 if x.astype('M8[D]').astype('O') in jp_holidays else 0 for x in array]  )


 
def to_float(v):
  try :
      return float(x)
  except :
      return 0.0




def todatetime2(x):
  try :  return arrow.get(x, "YYYYMMDD").naive
  except :   return 0


def todatetime(x) : return pd.to_datetime( str(x) )


def dd(x):  return pd.to_datetime(str(x))


import datetime 
def weekmonth(date_value):
     w = (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)
     if w < 0 or w > 6 :
         return -1
     else :
         return w

def weekyear2(dt) :
 return ((dt - datetime.datetime(dt.year,1,1)).days // 7) + 1    




   
def todatetime(x):
  try :  return arrow.get(x, "YYYYMMDD").naive
  except :   return 0


def weekyear2(dt) :
 return ((dt - datetime.datetime(dt.year,1,1)).days // 7) + 1    



def weekday_excel(x) :
 wday= arrow.get( str(x) , "YYYYMMDD").isocalendar()[2]    
 if wday != 7 : return wday+1
 else :    return 1
 


def weekyear_excel(x) :     
 dd= arrow.get( str(x) , "YYYYMMDD")
 wk1= dd.isocalendar()[1]

 # Excel Convention
 # dd0= arrow.get(  str(dd.year) + "0101", "YYYYMMDD")
 dd0_weekday= weekday_excel( dd.year *10000 + 101  )
 dd_limit= dd.year*10000 + 100 + (7-dd0_weekday+1) +1

 ddr= arrow.get( str(dd.year) + "0101" , "YYYYMMDD")
 # print dd_limit
 if    int(x) < dd_limit :
    return 1
 else :    
     wk2= 2 + int(((dd-ddr ).days  - (7-dd0_weekday +1 ) )   /7.0 ) 
     return wk2   
 
    
def datetime_generate(start='2018-01-01', ndays=100) :
 from dateutil.relativedelta import relativedelta   
 start0 = datetime.datetime.strptime(start, "%Y-%m-%d")
 date_list = [start0 + relativedelta(days=x) for x in range(0, ndays)]
 return date_list


def rmse( df, dfhat) :
  ss = np.sqrt(np.mean((dfhat.loc[:len(df), 'yhat'] - df['y'])**2)) 
  med =  df['y'].median()
  
  return (ss, med, ss / med)


def ffmat(x):
  if int(x) < 10 : return '0' + str(x)
  else :           return str(x)  

    
def merge1(x,y) :
   try :
      return int( ffmat(x) + ffmat(y)) 
   except : return -1     
    

    

def rmse( df, dfhat) :
  try :  
    ss  =  np.sqrt(np.mean((dfhat['yhat'].iloc[:len(df), : ] - df['y'])**2)) 
    med =  df['y'].median()
    return (ss, med, ss / med)
  except :
    ss  =  np.sqrt(np.mean((dfhat - df)**2))       
    med =  np.median( df )
    return (ss, med, ss / med)


  
    
########################################################################################################################    
def insert_batch_df(session,  tablename='ztest.ui_json2', prepared_query=None, df=None, cols_type=None,
                     zlib_compression_cols=None, consistency_level="QUORUM", concurrent=30, use_batch=0, mbatch=100, verbose=1) : 
    """
      Insert in batch
      Cassandra session      
      tablename = 'ztest.ui_json2'
      df : Pandas dataframe

      insert_batch_df(session,  tablename='ztest.item_master_uat', prepared_query=None, df=df, 
                consistency_level="QUORUM", cols_type=None, concurrent=30, use_batch=0, mbatch=3, verbose=1)               
    """
    from cassandra.concurrent import execute_concurrent_with_args
    from cassandra import ConsistencyLevel
    from cassandra.query import (PreparedStatement,  SimpleStatement,  BatchStatement, BatchType)

    cols   = list(df.columns)
    ncol   = len(cols)
    n      = len(df)

    if not cols_type is None :
      for i in range(len(cols)) :
         df[cols[i]] = df[cols[i]].astype(cols_type[i])
            
    #### query  ###################################################
    if prepared_query is None :
      level          = {"ONE" : ConsistencyLevel.ONE, "QUORUM": ConsistencyLevel.QUORUM, "ALL": ConsistencyLevel.ALL }
      cols           = tuple(cols)
      x              = ",".join(["?"] * ncol)
      cols_str       = ",".join(cols)
      query          = f"INSERT INTO  {tablename}  ({cols_str})  VALUES ({x})  ;  "
      prepared_query = session.prepare(query)
      prepared_query.consistency_level = level[consistency_level]
      if verbose : log(prepared_query)

    
    #### Add Mini Batch############################################
    # mbatch = 100
    mbatch      = mbatch if use_batch else concurrent
    batch_total = int(n // mbatch) + 1
    if verbose :
        log_time("total batch", batch_total, "Size ", mbatch)
    for i in range( batch_total ):
      values = df.iloc[i*mbatch:(i+1)*mbatch, :].values
      values = list(values)  #list of array
      if verbose : log_time(f"batch  {i}, Sending: {len(values)},  ")

      #### Execution 
      if use_batch :
        batch = BatchStatement(BatchType.LOGGED)
        for vec in  values:
            batch.add(prepared_query, vec)
        future = session.execute_async(batch)  ### Future results
       

      else : 
        if concurrent > 0 :      
          res = execute_concurrent_with_args(session, prepared_query, values, concurrency=concurrent)



def create_insert_query(session,  tablename='ztest.ui_json2', cols=None,  consistency_level="QUORUM", verbose=1) : 
      from cassandra import ConsistencyLevel
      ncol           = len(cols)
      level          = {"ONE" : ConsistencyLevel.ONE, "QUORUM": ConsistencyLevel.QUORUM, "ALL": ConsistencyLevel.ALL }
      cols           = tuple(cols)
      x              = ",".join(["?"] * ncol)
      cols_str       = ",".join(cols)
      query          = f"INSERT INTO  {tablename}  ({cols_str})  VALUES ({x})  ;  "
      prepared_query = session.prepare(query)
      prepared_query.consistency_level = level[consistency_level]
      return prepared_query

    
 ########################################################################################################################           
def cql_read_query_df(query= " SELECT  * FROM {table} WHERE key= '{key}' ",
                  session=None, consistency="ALL", timeout=None, verbose=1):
    """
        query and return dataframe
    """
    from cassandra import ConsistencyLevel
    from cassandra.query import (PreparedStatement, BoundStatement, SimpleStatement, BatchStatement)
    from time import time

    level = {"ONE": ConsistencyLevel.QUORUM, "QUORUM": ConsistencyLevel.QUORUM, "ALL": ConsistencyLevel.ALL}[
        consistency]

    # session = session_create() if session is None else session

    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)
    session.row_factory = pandas_factory
    session.default_fetch_size = None

    query2 = SimpleStatement(str(query), consistency_level=level)

    t0 = time()
    js = session.execute(query2, timeout=timeout)
    if verbose: 
        log(time()-t0, query2, js, )
        
    if not js._current_rows.empty:    
      df = js._current_rows
      return df 
    return None


########################################################################################################################
def to_file( txt="", filename="ztmp.txt",  mode='a'):
    with open(filename, mode=mode) as fp:
        fp.write(txt)
        
def date_now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M-%S")

        
def date_now_jp(fmt="%Y-%m-%d %H:%M:%S %Z%z", add_days=0):
    from pytz import timezone
    from datetime import datetime
    # Current time in UTC
    now_utc = datetime.now(timezone('UTC'))
    now_new = now_utc+ datetime.timedelta(days=add_days)

    # Convert to US/Pacific time zone
    now_pacific = now_new.astimezone(timezone('Asia/Tokyo'))
    return now_pacific.strftime(fmt)


####################################################################################################
def cql_keybuilder(key_dict):
    ss = ""
    for k, x in key_dict.items():
        ss = ss + f" {k}='{x}' "
    return ss


def cqlsh_execute(q, server_id=None, port=9042,     data_dir = "/tmp/"):
    # Details CQL
    if server_id is None :
        server_id =  ['scylladb102.analysis-shared.jpe2b.dcnw.rakuten',
                      'scylladb101.analysis-shared.jpe2b.dcnw.rakuten',
                      'scylladb103.analysis-shared.jpe2b.dcnw.rakuten'][ random.randrange(3)]
        
    os.makedirs(data_dir + "/ztmp/", exist_ok=True)
    tmp_file = data_dir + "/ztmp/cql_query_" + str(random.randrange(9999)) + ".sql"
    with open(tmp_file, mode='w') as fp:
        fp.write(q)
    cmd = f" cqlsh  {server_id} {port}  -f {tmp_file}  --connect-timeout 3600 --request-timeout 3600  "
    os.system(cmd)

    
################################################################################









