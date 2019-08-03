# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload 2







import os
from collections import OrderedDict

#### CatbOost
import catboost as cb
import lightgbm as lgb
import pandas as pd
import tensorflow as tf
from lightgbm.sklearn import LGBMClassifier, LGBMClassifiers
from mlens.ensemble import BlendEnsemble, SuperLearner
####################################################################################################
####################################################################################################
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
####Ensemble Learning
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, Imputer, OneHotEncoder, StandardScaler)
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper, cross_val_score



import da
####################################################################################################
################ Serialized the models #############################################################
import util
import util_feature
#from dask.multiprocessing import get
from util_feature import pd_col_to_num, pd_col_to_onehot, pd_colnum_tocat
import util_model 
import util_plot 
import util_text 
import util_date 



dir0 = os.getcwd()
os.chdir( dir0 + "/da/")
print( os.getcwd() )

print(tf, tf.__version__)


util_text.








""""
### Framework of pandas and PySpark are close, easier to streamline in p
Functionnal appraoch is easier to streamline for Spark, PySpark, pre-processing.
Scala too,

sklearn is class based transformer... while benefit, much difficut to 
transpose into Spark processing.

Functionnal programming is better for Serialization 
(ie class has issues with pickle and distributed compute)


Problem is with Spark , we cannot use efficiently scikit learn...
   --> We need to work with mapping dictionnary



"""

##### For Merge dataframe
df  =  df.set_index("id")



##### Numerics pre-processing 
df_num = pd_to_num(df[colnum], default= np.nan )


##### float to category bins
df_num, colnum_binmap = pd_colnum_tocat(
    df_num, colname=None, colexclude=None, colbinmap=None, bins=5, suffix="_bin", method="uniform",
    return_val="dataframe,param"
    )

        
### numerics category bin to One Hot
dfnum, colnum_onehot = pd_col_to_onehot(df_num, colname=None, colonehot=None,  
                                        return_val="dataframe,param")
dfnum.head(5)


### Reproductible pipeline (can be easily re-coded for Spark !!!) ##################################
pipe_preprocess_colnum =[ 
           (pd_col_to_num,   {"default": np.nan,} , "Conver string to NA")
           
          ,(pd_colnum_tocat, { "colname":None, "colbinmap": colnum_binmap,  'bins': 5, 
                               "method": "uniform", "suffix":"_bin", "return_val": "dataframe"}, 
                               "Convert Numerics to Category " )
           
          ,(pd_col_to_onehot, { "colname": None,  "colonehot": colnum_onehot,  
                                "return_val": "dataframe"  } , 
                                "Convert category to onehot" )
]



### Reproductible pipeline (can be easily re-coded for Spark !!!) ##################################
pipe_preprocess_colnum =[ 
           (pd_col_to_num,   {"default": np.nan,} , "Conver string to NA")
           
          ,(pd_colnum_tocat, { "colname":None, "colbinmap": colnum_binmap,  'bins': 5, 
                               "method": "uniform", "suffix":"_bin", "return_val": "dataframe"}, 
                               "Convert Numerics to Category " )
           
]


pipe_preprocess_colnum

dfnum_test = pd_pipeline_apply( df[colnum], pipe_preprocess_colnum)  






####################################################################################################
###### Category pre-processing  ####################################################################
#### Replace NA values by flag value "-1"
df_cat, map_navalue_cat = pd_col_fillna(df[colcat], method="", value="-1" )

### String cat to integer cat mapping
colcat_catmap = pd_colcat_mapping(df_cat, colname= list(df_cat.columns) ) 

  
### Category cat to One Hot Category
df_cat, colcat_onehot = pd_col_to_onehot(df_cat, colname=None, 
                                         colonehot=None,  return_val="dataframe,param")
df_cat.head(5)


### Reproductible pipeline (can be easily re-coded for Spark !!!)
pipe_preprocess_colcat =[ 
           (pd_col_fillna, {"value": "-1", "method" : "",   "return_val": "dataframe" },
           )        

          ,(pd_col_to_onehot, { "colname": None,  "colonehot": map_colcat_onehot,  
                                "return_val": "dataframe"  } , "convert to one hot")
]




####### Merge dataframe
dftrain = pd.concat((  df_num, df_cat  ) , axis=1)


coltext








from  column_encoder import MinHashEncoder


#### Output Vector Dimension
enc =  MinHashEncoder(2)
enc.fit( df['phone_equality']  )


vv = enc.transform( df['phone_equality'].values  )
df2 = pd.DataFrame( vv )

df2["cat0"] = df['phone_equality'].reset_index().iloc[:,1]


dfs = df2.groupby("cat0"  ).agg({ "cat1" :  {"min", "max"}  })




df2




def pd_coltext_minhash(dfref, colname, n_component=2, model_pretrain_dict=None, return_val="dataframe,param"):
    """
    dfhash, colcat_hash_param = pd_colcat_minhash(df, colcat, n_component=[2] * len(colcat),
                                              return_val="dataframe,param")
    :param dfref:
    :param colname:
    :param n_component:
    :param return_val:
    :return:
    """
    df = dfref[colname]
    model_pretrain_dict= {}  if model_pretrain_dict is None else model_pretrain_dict
    enc_dict = {}
    for i, col in enumerate(colname):
        
        if model_pretrain_dict.get(col) is None :
           clf = MinHashEncoder(n_component[i])
           clf = clf.fit(df[col])
        else :
           clf = copy.deepcopy( model_pretrain_dict[col] )  
            
        v = clf.transform(df[col].values)
        
        enc_dict[col] = copy.deepcopy(clf)
        dfcat = pd.DataFrame(v, columns=["{col}_hash_{t}".format(col=col, t=t) for t in
                                         range(0, v.shape[1])])

        try:
            dfall = pd.concat((dfall, dfcat), axis=1)
        except:
            dfall = dfcat

    if return_val == "dataframe,param":
        return dfall, enc_dict

    else:
        return dfall




import util_plot
util_plot.plotxy( dfhash.iloc[ :, -2], dfhash.iloc[ :, -1]   )


util_text.


dfdate_hash, coldate_hash_model= pd_coltext_minhash(df, coldate, n_component=[4, 2], 
                                               model_pretrain_dict=None,       
                                              return_val="dataframe,param") 




######### Pipeline ##########################################
pipe_preprocess_coldate_01 =[ 
           (pd_coltext_minhash , {"colname": coldate, "n_component" : [],
                                          "model_pretrain_dict" : coldate_hash_model,
                                           "return_val": "dataframe"  },  )        
     
]

    
    
### Check pipeline
print( coldate )
util_feature.pd_pipeline_apply( df[ coldate ].iloc[:10, :], pipe_preprocess_coldate_01).iloc[:10]  





coldate = ['first_review', 'host_since']




df[coldate] = df[coldate].fillna("")

df[coldate].nunique()


df[]






      
####################################################################################################         
####################################################################################################
from sklearn.decomposition import PCA, pca, TruncatedSVD,LatentDirichletAllocation, NMF
    
    

def pd_dim_reduction(df, colname, colprefix="colsvd", method="svd", dimpca=2, whiten=True,
                     return_val="dataframe,param"):
    """
       Into Smaller space
       dftext_svd, svd = pd_dim_reduction(dfcat_test, None,colprefix="colsvd",
                     method="svd", dimpca=2, return_val="dataframe,param")

    :param df:
    :param colname:
    :param colprefix:
    :param method:
    :param dimpca:
    :param whiten:
    :param return_val:
    :return:
    """
    colname = colname if colname is not None else list(df.columns)
    if method == "svd" :
      svd = TruncatedSVD( n_components= dimpca, algorithm='randomized')
      svd = svd.fit( df[colname].values )
      X2 = svd.transform(df[colname].values)
      print(X2)
      dfnew = pd.DataFrame(X2)
      dfnew.columns = [ colprefix + "_" +str(i) for i in dfnew.columns  ]

      if return_val == "dataframe,param" :
        return dfnew, svd
      else :
        return dfnew
    
    
col = 'house_rules'
dftext_svd_list, svd_list = pd_dim_reduction(dftext_tdidf[col], None,colprefix="colsvd",
                     method="svd", dimpca=2, return_val="dataframe,param")





dftext_svd_list, svd_list = pd_dim_reduction(dfcat_test, None,colprefix="colsvd",
                     method="svd", dimpca=2, return_val="dataframe,param")




        
def pd_coltext_tdidf(df, coltext, word_tokeep=None, word_minfreq=1,
                     return_val="dataframe,param"):
    """
    Function that adds tf-idf of a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        col_tofilter:   column of df to apply tf-idf to
    Returns:
        concat_df:      dataframe with a new column for each word
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    from sklearn.feature_extraction.text import CountVectorizer
    if not isinstance(coltext, str) :
        raise "coltext should be column string"

    if word_tokeep is None:
        cv = CountVectorizer(min_df=1, ngram_range=(1,3),
                       strip_accents='unicode',
                       lowercase =True, analyzer='word', token_pattern=r'\w+',
                       stop_words = None)
        X = cv.fit_transform(df[coltext])
        word_tokeep = cv.get_feature_names()
        count_list = np.asarray(X.sum(axis=0))
        word_dict = dict(zip(word_tokeep, count_list))
        print(len(word_tokeep))

    # Calculate td-idf vector
    vectorizer = TfidfVectorizer()
    vectorizer.fit(word_tokeep)
    v = vectorizer.transform(df[col])
    v = v.toarray()
    print(v.shape)

    voca = vectorizer.vocabulary_

    df_vector = pd.DataFrame(v)
    # df_new = pd.concat([df, df_vector],axis=1)
    if return_val == "dataframe,param":
        return df_vector, voca

    else:
        return df_vector

    
import string
import util_text

def pd_coltext_clean(dfref, colname, stopwords):
    df = dfref[colname]
    # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]  
    # print(fromword)
    for col in colname:
        df[col] = df[col].fillna("")
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: x.translate(string.punctuation))
        df[col] = df[col].apply(lambda x: x.translate(string.digits))
        df[col] = df[col].apply(lambda x: re.sub("[!@,#$+%*:()'-]", ' ', x))

        df[col] = df[col].apply(lambda x: coltext_stopwords(x, stopwords=stopwords ))

    return df


df1 


stopwords = ['', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '{', '|', '}', '~', '€']



def coltext_stopwords(text, stopwords=None, sep=" "):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split(" ")
    stemmed_tokens = [t.strip() for t in tokens if t.strip() not in stopwords ]
    return " ".join(stemmed_tokens)




## To pipeline
#  Save each feature processing into "Reproductible pipeline".
#  For Scalability, for repreoduction process
ntoken=20
col = 'house_rules'


word_tokeep = coltext_freq[col]["word"].values[:ntoken]
word_tokeep = [  t for t in word_tokeep if t not in stopwords   ]



word_tokeep =  ['please', 'die', 'und', 'apartment', 'smoking', 'der', 'check', 'bitte', 'ist', 'zu', 'use', '00', 'house', 'wohnung', '5', 'für', 'sie', 'nicht', 'das']



col = 'summary'

pipe_preprocess_coltext01 =[ 
           ( pd_coltext_clean , {"colname": [col], "stopwords"  : stopwords },  )        

          ,( pd_coltext_tdidf, { "coltext": col,  "word_minfreq" : 1,
                                          "word_tokeep" : word_tokeep ,
                                          "return_val": "dataframe"  } , "convert to TD-IDF vector")
]



pipe_preprocess_coltext01 =[ 
           ( pd_coltext_clean , {"colname": col, "stopwords"  : stopwords },  )        

]

    

print( col, word_tokeep )
### Check pipeline


util_feature.pd_pipeline_apply( dftext.iloc[:100,:], pipe_preprocess_coltext01)  



string.punctuation


folder = os.getcwd() + "/data/airbnb/"
df = pd.read_csv(folder+'listings_summary.zip', delimiter=',')




colid = "id"
colnum = [  "review_scores_communication", "review_scores_location", "review_scores_rating"      ]

colcat = [ "cancellation_policy", "host_response_rate", "host_response_time" ]

coltext = ["house_rules", "neighborhood_overview", "notes", "street"  ]

coldate = [  "calendar_last_scraped", "first_review", "host_since" ]


coly = "price"


coltext = [ "summary", "space" ]

dftext = df[coltext ]



dftext[  "house_rules"  ]

    
    


df.columns



####################################################################################################
#################### Pipeline for production #######################################################
#colnum_binmap = None
# colnum_onehot = None

def pd_pipeline_apply(df, pipeline) :            
  dfi = copy.deepcopy( df )   
  for i, function in enumerate( pipeline) :
     print("############## Pipeline ", i, "Start", dfi.shape, function[0].__name__, flush=True )
     dfi = function[0]( dfi, **function[1] )    
     print("############## Pipeline  ", i, "Finished",  dfi.shape, flush=True )
  return dfi





#### colnum pre-process     ####################################
dfnum_test = pd_pipeline_apply( df[colnum], pipe_preprocess_colnum)  



############# Colcat preprocess  ##############################
dfcat_test = pd_pipeline_apply( df[colcat], pipe_preprocess_colcat )  



#### Merge Dataframe
dftest = pd.concat(( dfnum_test, dfcat_test ), axis= 1)










colname_list  =  [  "colnum", "colnum_bin",  
                  "colcat", "colcat_onehot" ] 
colname_list  = { x: globals()[x]   for x in colname_list }


folder_model = folder + "/models/model/" 


var_list = [
     "pipe_preprocess_colnum",  ## Pre-processors
     "pipe_preprocess_colcat",  ##
     
     ##### Data
     "df",
     "df_cat",
     "df_num",
     "df_train"
     "colname_list",
      
     ##### Model
     "clf_log",
     "clf_lgb",
     "clf_svc"
        
     ]

util.save_all(var_list , folder_model, globals_main= globals() ) 
 
 
##### Check
pipe_preprocess_colnum2 = util.load( folder + "/models/model/pipe_preprocess_colnum.pkl")

dfnum_test = pd_pipeline_apply( df[colnum], pipe_preprocess_colnum2 )  







def load(filename):
    ...

def clean(data):
    ...

def analyze(sequence_of_data):
    ...

def store(result):
    with open(..., 'w') as f:
        f.write(result)

dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}

get(dsk, 'store')  # executes in parallel

http://ml-ensemble.com/info/







###### Technically we need to setup a compute graph   #############################################
###  Compute Graph  : Data  and model  (ie DAG in )

>>> from __future__ import print_function

>>> def print_and_return(string):
...     print(string)
...     return string

>>> def format_str(count, val, nwords):
...     return ('word list has {0} occurrences of {1}, '
...             'out of {2} words').format(count, val, nwords)

>>> dsk = {'words': 'apple orange apple pear orange pear pear',
...        'nwords': (len, (str.split, 'words')),
...        'val1': 'orange',
...        'val2': 'apple',
...        'val3': 'pear',
...        'count1': (str.count, 'words', 'val1'),
...        'count2': (str.count, 'words', 'val2'),
...        'count3': (str.count, 'words', 'val3'),
...        'out1': (format_str, 'count1', 'val1', 'nwords'),
...        'out2': (format_str, 'count2', 'val2', 'nwords'),
...        'out3': (format_str, 'count3', 'val3', 'nwords'),
...        'print1': (print_and_return, 'out1'),
...        'print2': (print_and_return, 'out2'),
...        'print3': (print_and_return, 'out3')}









colcat_idx = col_getnumpy_indice(colall, colcat)

clf_cb = cb.CatBoostClassifier( iterations=1000, depth=8,
                          learning_rate=0.02, loss_function='Logloss',
                          eval_metric='AUC', random_seed=42,
                          rsm = 0.2, # features subsample 
                          od_type = 'Iter', # early stopping odwait = 100, # early stopping
                          verbose = 100,l2_leaf_reg = 20, # regularisation 
)




def np_find_indice(v, x) :
  for i, j in enumerate(v):
    if j == x :
       return i
  return -1




def col_getnumpy_indice(colall, colcat) :
    return [  np_find_indice(colall, x)  for x in colcat ]



#clf_cb, dd_cb = sk_model_eval_classification(clf_cb, 1,
#                                               Xtrain, ytrain, Xtest, ytest)


clf_cb.fit(Xtrain, ytrain,
                 eval_set=(Xtest, ytest),
                 cat_features= np.arange(0, Xtrain.shape[1]),
                 use_best_model=True)









# splitx() is in a separate module, to make it available to pickled pipelines
def splitx(X, numlen):
    """ Split 2D np.array (observations x features) into a list of arrays.
    First all numeric features together, assume all numerics are in beginning cols.
    Then an array for each categorical feature, 1 col each.
    Used in Keras models with embeddings and multiple inputs.
    """
    L = [X[:, :numlen]]
    for i in range(numlen, X.shape[1]):
        L.append(X[:, i])
    return L

splitx_ft = FunctionTransformer(splitx, validate=True, 
                                kw_args={'numlen': len(numeric_atts)})

num_pipeline = Pipeline([
    ('num_selector', DataFrameSelector(numeric_atts)),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('cat_selector', DataFrameSelector(category_atts)),
    ('ordinal_encoder', make_cat_encoder(category_atts, encoding='ordinal', handle_unknown='error'))
])

fu_pipeline = FeatureUnion([
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

pipeline = Pipeline([
    ('fu_pipeline', fu_pipeline),
    ('splitx_ft', splitx_ft)
])



    
    

def all_but_first_column(X):
    return X[:, 1:]


def drop_first_component(X, y):
    """
    Create a pipeline with PCA and the column selector and use it to
    transform the dataset.
    """
    pipeline = make_pipeline(
        PCA(), FunctionTransformer(all_but_first_column),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    return pipeline.transform(X_test), y_test




mapper = DataFrameMapper([
...     ('pet', sklearn.preprocessing.LabelBinarizer()),
...     (['children'], sklearn.preprocessing.StandardScaler())
... ])
    
>>> import sklearn.preprocessing, sklearn.decomposition, \
...     sklearn.linear_model, sklearn.pipeline, sklearn.metrics
>>> from sklearn.feature_extraction.text import CountVectorizer



mapper_alias = DataFrameMapper([
...     (['children'], sklearn.preprocessing.StandardScaler(),
...      {'alias': 'children_scaled'})
... ])








class ColumnSelector(BaseEstimator, TransformerMixin):
    ### Select Subset of columns from dataframe
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    ### Select Subset of columns from dataframe
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


        ("colcat", make_pipeline(
            ColumnSelector(columns= colcat),
            to_na(val="?"),
            Imputer(strategy="most_frequent"),
            OneHotEncoder()
        )),
        
        
        

preprocess_pipeline = make_pipeline(
    FeatureUnion(transformer_list=[
            
        ("colnum", make_pipeline(
            to_na(val="?"),    
            ColumnSelector(columns= colnum),
            Imputer(strategy="median"),
            StandardScaler()
            
        )),

    ])
)



normalize = make_pipeline(
    FunctionTransformer(np.nan_to_num, validate=False),
    Normalize()
)


def pd_to_na(df, val="?") :
    return df.replace(val, np.nan)
    


to_na = FunctionTransformer( pd_to_na, validate=False, 
                                    kw_args = {"val": "?" } )


to_na.fit_transform(df)



    



dfnum = preprocess_pipeline.fit_transform(df[colall])    
    



mapper_df = DataFrameMapper([        
     (colnum, [ to_na ,
                sk.preprocessing.StandardScaler() ]
            ,  {'input_df': True}
     )
 ], df_out=True)




###### SKlearnize function 
colnum_tocat = FunctionTransformer( pd_colnum_tocat, validate=False, 
                                    kw_args = {"colname":None, 'bins': 5, "method": "uniform",
                                               "suffix":"_bin", "return_val": "dataframe"} )

colnum_tocat.fit_transform( df[colnum].replace("?", np.nan) )



mapper_df = DataFrameMapper([        
     (colnum, [ to_na ,
                colnum_tocat ]
            ,  {'input_df': True}
     )
 ], df_out=True)


dfnum_bin= mapper_df.fit_transform( df[colnum]) 







    
preprocess_pipeline = make_pipeline(
    ColumnSelector(columns= colall),
    FeatureUnion(transformer_list=[
            
        ("colnum", make_pipeline(
            ColumnSelector(columns= colnum),
            Imputer(strategy="median"),
            StandardScaler()
        )),
        
        ("colcat", make_pipeline(
            ColumnSelector(columns= colcat),
            Imputer(strategy="most_frequent"),
            OneHotEncoder()
        )),
        
    ])
)











param_grid = {
    "svc__gamma": [0.1 * x for x in range(1, 6)]
}

classifier_model = GridSearchCV(classifier_pipeline, param_grid, cv=10)
classifier_model.fit(X_train, y_train)







################################################################################################        
################################################################################################    
def pd_colnum_tocat(
    df, colname=None, colexclude=None, colbinmap=None, bins=5, suffix="_bin", method="uniform"
):




classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    SVC(kernel="rbf", random_state=42)
)


>>> transformers = [('cat', cat_pipe, cat_cols),
                    ('num', num_pipe, num_cols)]
>>> ct = ColumnTransformer(transformers=transformers)
>>> X = ct.fit_transform(train)
>>> X.shape
(1460, 305)




###########################################################################
############ Create Pipeline Design #######################################    
model_pipeline.fit(train_data.values, train_labels.values)
predictions = model_pipeline.predict(predict_data.values)





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
