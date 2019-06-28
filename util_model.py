"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""


def split_train(df1, ntrain=10000, ntest=100000, colused=None ) :
    pass


def split_train2(df1, ntrain=10000, ntest=100000, colused=None, nratio =0.4 ) :
    pass



    
######################  Category Classifier Trees  #########################################################################
"""
Category Classifier
https://github.com/catboost/catboost/blob/master/catboost/tutorials/kaggle_paribas.ipynb

Very Efficient
D:\_devs\Python01\project27\linux_project27\mlearning\category_learning


https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/


clf = CatBoostClassifier(learning_rate=0.1, iterations=1000, random_seed=0)
clf.fit(train_df, labels, cat_features=cat_features_ids)


##### Base Approach
import pandas as pd
import numpy as np

from itertools import combinations
from catboost import CatBoostClassifier


labels = train_df.target
test_id = test_df.ID

train_df.drop(['ID', 'target'], axis=1, inplace=True)
test_df.drop(['ID'], axis=1, inplace=True)

train_df.fillna(-9999, inplace=True)
test_df.fillna(-9999, inplace=True)

# Keep list of all categorical features in dataset to specify this for CatBoost
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()



########  Regularizer
selected_features = [
    'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50',
    'v52', 'v56', 'v62', 'v66', 'v72', 'v75', 'v79', 'v91', 'v112', 'v113', 'v114', 'v129'
]

# drop some of the features that were not selected
train_df = train_df[selected_features]
test_df = test_df[selected_features]

# update the list of categorical features
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()


char_features = list(train_df.columns[train_df.dtypes == np.object])
char_features_without_v22 = list(train_df.columns[(train_df.dtypes == np.object) & (train_df.columns != 'v22')])

cmbs = list(combinations(char_features, 2)) + map(lambda x: ("v22",) + x, combinations(char_features_without_v22, 2))


clf = CatBoostClassifier(learning_rate=0.1, iterations=1000, random_seed=0)
clf.fit(train_df, labels, cat_features=cat_features_ids)


"""


def sk_catboost_classifier(
    Xtrain,
    Ytrain,
    Xcolname=None,
    pars={
        "learning_rate": 0.1,
        "iterations": 1000,
        "random_seed": 0,
        "loss_function": "MultiClass",
    },
    isprint=0,
):
    """
  from catboost import Pool, CatBoostClassifier

TRAIN_FILE = '../data/cloudness_small/train_small'
TEST_FILE = '../data/cloudness_small/test_small'
CD_FILE = '../data/cloudness_small/train.cd'
# Load data from files to Pool
train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
test_pool = Pool(TEST_FILE, column_description=CD_FILE)
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass')
# Fit model
model.fit(train_pool)
# Get predicted classes
preds_class = model.predict(test_pool)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_pool)
# Get predicted RawFormulaVal
  preds_raw = model.predict(test_pool, prediction_type='RawFormulaVal')


  https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/

  """
    pass



def sk_catboost_regressor():
    pass


######################  ALGO  #########################################################################
def sk_model_auto_tpot(
    Xmat,
    y,
    outfolder="aaserialize/",
    model_type="regressor/classifier",
    train_size=0.5,
    generation=1,
    population_size=5,
    verbosity=2,
):
    """ Automatic training of Xmat--->Y, Generate SKlearn code in outfile
      Very Slow Process, use lower number of Sample
  :param Xmat:
  :param y:
  :param outfolder:
  :param model_type:
  :param train_size:
  :param generation:
  :param population_size:
  :param verbosity:
  :return:
  """

    pass
    


def sk_params_search_best(
    Xmat,
    Ytarget,
    model1,
    param_grid={"alpha": np.linspace(0, 1, 5)},
    method="gridsearch",
    param_search={"scoretype": "r2", "cv": 5, "population_size": 5, "generations_number": 3},
):
    pass  

    """
   genetic: population_size=5, ngene_mutation_prob=0.10,,gene_crossover_prob=0.5, tournament_size=3,  generations_number=3

  :param Xmat:
  :param Ytarget:
  :param model1:
  :param param_grid:
  :param method:
  :param param_search:
  :return:
  """

    """
   from sklearn.metrics import  make_scorer,  r2_score
from sklearn.grid_search import GridSearchCV

myscore = make_scorer(r2_score, sample_weight=None)

param_grid= {'alpha': np.linspace(0.01, 1.5, 10),
             'ww': [[0.05, 0.95]],
             'low_y_cut': [-10.0], 'high_y_cut': [9.0]   }

grid = GridSearchCV(model1(),param_grid, cv=10, scoring=myscore) # 20-fold cross-validation
grid.fit(Xtrain, Ytrain)
grid.best_params_

# Weight Search
wwl= np.linspace(0.01, 1.0, 5)
param_grid= {'alpha':  [0.01],
             'ww0': wwl,
             'low_y_cut': [-0.08609*1000], 'high_y_cut': [0.09347*1000]   }

grid = GridSearchCV(model1(),param_grid, cv=10, scoring=myscore) # 20-fold cross-validation
grid.fit(X*100.0, Ytarget*1000.0)
grid.best_params_

# {'alpha': 0.01, 'high_y_cut': 93.47, 'low_y_cut': -86.09, 'ww0': 0.01}

   """


def sk_distribution_kernel_bestbandwidth(kde):
    """Find best Bandwidht for a  given kernel
  :param kde:
  :return:
 """
    pass


def sk_distribution_kernel_sample(kde=None, n=1):
    """
  kde = sm.nonparametric.KDEUnivariate(np.array(Y[Y_cluster==0],dtype=np.float64))
  kde = sm.nonparametric.KDEMultivariate()  # ... you already did this
 """
    pass


def sk_correl_rank(correl=[[1, 0], [0, 1]]):
    """ Correl Ranking:  Col i, Col j, Correl_i_j, Abs_Correl_i_j    """
    pass


def sk_error_r2(Ypred, y_true, sample_weight=None, multioutput=None):
    pass


def sk_error_rmse(Ypred, Ytrue):
    pass


def sk_cluster_distance_pair(Xmat, metric="jaccard"):
    """
    'euclidean, 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean, 'cosine, 'correlation, 'hamming, 'jaccard, 'chebyshev, 'canberra, 'braycurtis, 'mahalanobis', VI=None) 'yule, 'matching, 'dice, 'kulsinski, 'rogerstanimoto, 'russellrao, 'sokalmichener, 'sokalsneath,

    'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}
   #Visualize discretization scheme

   Xtrain_dist= sci.spatial.distance.squareform(sci.spatial.distance.pdist(Xtrain_d,
             metric='cityblock', p=2, w=None, V=None, VI=None))

   Xtsne= da.plot_cluster_tsne(Xtrain_dist, metric='', perplexity=40, ncomponent=2, isprecompute=True)

   """
    pass


def sk_cluster(
    Xmat,
    method="kmode",
    args=(),
    kwds={"metric": "euclidean", "min_cluster_size": 150, "min_samples": 3},
    isprint=1,
    preprocess={"norm": False},
):
    pass
    """
   'hdbscan',(), kwds={'metric':'euclidean', 'min_cluster_size':150, 'min_samples':3 }
   'kmodes',(), kwds={ n_clusters=2, n_init=5, init='Huang', verbose=1 }
   'kmeans',    kwds={ n_clusters= nbcluster }

   Xmat[ Xcluster== 5 ]
   # HDBSCAN Clustering
   Xcluster_hdbscan= da.sk_cluster_algo_custom(Xtrain_d, hdbscan.HDBSCAN, (),
                  {'metric':'euclidean', 'min_cluster_size':150, 'min_samples':3})

   print len(np.unique(Xcluster_hdbscan))

   Xcluster_use =  Xcluster_hdbscan

# Calculate Distribution for each cluster
kde= da.plot_distribution_density(Y[Xcluster_use== 2], kernel='gaussian', N=200, bandwith=1 / 500.)
kde.sample(5)

   """


def sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval=1):
    pass
    """ Plot the cLuster using specific Algo
    distance_matrix = pairwise_distances(blobs)
    clusterer = hdbscan.HDBSCAN(metric='precomputed')
    clusterer.fit(distance_matrix)
    clusterer.labels_

    {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}

    """


"""
def sk_cluster_kmeans(Xmat, nbcluster=5, isprint=False, isnorm=False) :
  from sklearn.cluster import k_means
  stdev=  np.std(Xmat, axis=0)
  if isnorm  : Xmat=   (Xmat - np.mean(Xmat, axis=0)) / stdev

  sh= Xmat.shape
  Xdim= 1 if len(sh) < 2 else sh[1]   #1Dim vector or 2dim-3dim vector
  print(len(Xmat.shape), Xdim)
  if Xdim==1 :  Xmat= Xmat.reshape((sh[0],1))

  kmeans = sk.cluster.KMeans(n_clusters= nbcluster)
  kmeans.fit(Xmat)
  centroids, labels= kmeans.cluster_centers_,  kmeans.labels_

  if isprint :
   import matplotlib.pyplot as plt
   colors = ["g.","r.","y.","b.", "k."]
   if Xdim==1 :
     for i in range(0, sh[0], 5):  plt.plot(Xmat[i], colors[labels[i]], markersize = 5)
     plt.show()
   elif Xdim==2 :
     for i in range(0, sh[0], 5):  plt.plot(Xmat[i,0], Xmat[i,1], colors[labels[i]], markersize = 2)
     plt.show()
   else :
      print('Cannot Show higher than 2dim')

  return labels, centroids, stdev
"""


def sk_optim_de(obj_fun, bounds, maxiter=1, name1="", solver1=None, isreset=1, popsize=15):
    """ Optimization and Save Data into file"""
    pass

######## Valuation model template  ##########################################################
class sk_model_template1(sk.base.BaseEstimator):
    def __init__(self, alpha=0.5, low_y_cut=-0.09, high_y_cut=0.09, ww0=0.95):
        pass
    def fit(self, X, Y=None):
        pass
    def predict(self, X, y=None, ymedian=None):
        pass
    def score(self, X, Ytrue=None, ymedian=None):
        pass

############################################################################
# ---------------------             ----------------
"""
 Reshape your data either using X.reshape(-1, 1) if your data has a single feature or
  X.reshape(1, -1) if it contains a single sample.

"""


def sk_feature_importance(clfrf, feature_name):
    pass



# -------- SK Learn TREE UTIL----------------------------------------------------------------
def sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1=1, njobs=1):
    pass

def sk_gen_ensemble_weight(vv, acclevel, maxlevel=0.88):
    pass

def sk_votingpredict(estimators, voting, ww, X_test):
    pass

def sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base=" "):
    """Produce psuedo-code for decision tree.
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (output) names.
    spacer_base -- used for spacing code (default: "    ").
    """
    pass

"""
class META_DB_CLASS(object):
   # Create Meta database to store infos on the tables : csv, zip, HFS, Postgres
ALL_DB['japancoupon']= {}
ALL_DB['japancoupon']['schema']=    df_schema
ALL_DB['japancoupon']['table_uri']= df_schema
ALL_DB['japancoupon']['table_columns']= df_schema


   def __init__(self, db_file='ALL_DB_META.pkl') :
     if db_file.find('.pkl') != -1 :
      self.filename= db_file
      self.db= util.load(db_file, isabsolutpath=1)

   def db_add(self, dbname ):
     self.db[dbname]= {}    # util.np_dictordered_create()

   def db_update_item(self, dbname, itemlistname='table_uri/schema/table_columns', itemlist=[]):
     self.db[dbname][itemlistname]=  itemlist

   def db_save(self, filename='') :
     if filename== '' :
        util.save(self.db, self.filename, isabsolutpath=1)
     else :
        self.filename= filename
        util.save(self.filename)

   def db_print_item(self):
       pass

meta_db= META_DB_CLASS( in1+'ALL_DB_META.pkl')

"""


# ---------------------Execute rules on State Matrix --------------------------------
class sk_stateRule:
    """ Calculate Rule(True/False) based on State and Trigger
      Allow to add function to class externally                     """

    def __init__(self, state, trigger, colname=[]):
        pass
    def addrule(self, rulefun, name="", desc=""):
        pass
    def eval(self, idrule, t, ktrig=0):
        pass
    def help(self):
        """
s1= np.arange(5000).reshape((1000, 5))
trig1= np.ones((1,5))
state1= sk_stateRule(aa, trig1, ['drawdown','ma100d','ret10d','state_1','state_2'] )

def fun1(s, tr,t):
  return  s.drawdown[t] < tr.drawdown[0] and  s.drawdown[t] < tr.drawdown[0]

def fun2(s, tr,t):
 return  s.drawdown[t] > tr.drawdown[0] and  s.drawdown[t] < tr.drawdown[0]

state1.addrule(fun1, 'rule6')
state1.addrule(fun2, 'rule5')

state1.eval(idrule=0,t=5)

state1.eval(idrule=1,t=5)

state1.eval(idrule='rule5',t=6)

util.save_obj(state1, 'state1')

np.shape(aa2)

aa2= util.np_torecarray(aa,  ['drawdown','a2','a3','a4','a5'])

util.find(5.0, aa2[0])

recordarr = np.rec.array([(1,2.,7),(2,3.,5)],
                   dtype=[('col1', 'f8'),('col2', 'f8'), ('col3', 'f8')])
recordarr.col3[0]

state1= stateRule(np.ones((100,10)), np.ones((1,10)))

col= aa2.a2

"""


"""

def (X):
    return X[:, 1:]

def drop_first_component(X, y):
    "" Create a pipeline with PCA and the column selector and use it to transform the dataset. ""
    pipeline = make_pipeline( PCA(), FunctionTransformer(all_but_first_column))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    return pipeline.transform(X_test), y_test


"""


############################################################################
# ---------------------             --------------------
"""
Symbolic Regression:

http://gplearn.readthedocs.io/en/latest/examples.html#example-2-symbolic-tranformer


!pip install gplearn


x0 = np.arange(-1, 1, 1/10.)
x1 = np.arange(-1, 1, 1/10.)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                       color='green', alpha=0.5)
plt.show()

import gplearn as gp

rng = gp.check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

"""

# ------¨Pre-Processors --------------------------------------------------------------
"""
One-Hot: one column per category, with a 1 or 0 in each cell for if the row contained that column’s category
Binary: first the categories are encoded as ordinal, then those integers are converted into binary code,
then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot,
 but with some distortion of the distances.

http://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html

import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...])
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])

Best is Binary Encoder

Splice
Coding	Dimensionality	Avg. Score	Elapsed Time
14	Ordinal	61	0.68	5.11
17	Sum Coding	3465	0.92	25.90
16	Binary Encoded	134	0.94	3.35
15	One-Hot Encoded	3465	0.95	2.56


Value ---> Hash  (limited in value)
      ---> Reduce Dimensionality of the Hash

def hash_fn(x):
tmp = [0for_inrange(N)]
for val in x.values:
tmp[hash(val)% N] += 1
return pd.Series(tmp, index=cols)

cols = ['col_%d'% d for d in range(N)]
X = X.apply(hash_fn, axis=1)


@profile(precision=4)
def onehot():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.OneHotEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out

def binary(X):
    enc = ce.BinaryEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out

enc = ce.OneHotEncoder()
X_bin = enc.fit_transform(X)

import matplotlib.pyplot as plt
import category_encoders as ce
from examples.source_data.loaders import get_mushroom_data, get_cars_data, get_splice_data


"""

def sk_showmetrics(y_test, ytest_pred, ytest_proba,target_names=['0', '1']):
    pass



def sk_showmetrics2(y_test, ytest_pred, ytest_proba,target_names=['0', '1']) :
    pass


def clf_prediction_score(clf, df1 , cols, outype='score') :
    pass

def sk_showconfusion(clfrf, X_train, Y_train, isprint=True):
    pass
