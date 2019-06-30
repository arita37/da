# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""
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

import scipy as sci
import sklearn as sk
import statsmodels as sm
from dateutil.parser import parse
from sklearn import covariance, linear_model, model_selection
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from attrdict import AttrDict as dict2

# from kmodes.kmodes import KModes
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

try:
    from catboost import CatBoostClassifier, Pool, cv
except Exception as e:
    print(e)


#########################################################################################################
# DIRCWD= os.environ["DIRCWD"]; os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
# import configmy; CFG, DIRCWD= configmy.get(config_file="_ROOT", output= ["_CFG", "DIRCWD"])
DIRCWD = "./"

##############################################################################


def np_transform_pca(Xmat, dimpca=2, whiten=True):
    """Project ndim data into dimpca sub-space  """
    pca = PCA(n_components=dimpca, whiten=whiten).fit(Xmat)
    return pca.transform(Xmat)


def sk_feature_impt_logis(clf, cols2):
    dfeatures = pd.DataFrame(
        {"feature": cols2, "coef": clf.coef_[0], "coef_abs": np.abs(clf.coef_[0])}
    ).sort_values("coef_abs", ascending=False)
    dfeatures["rank"] = np.arange(0, len(dfeatures))
    return dfeatures


def sk_feature_importance(clfrf, feature_name):
    importances = clfrf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(0, len(feature_name)):
        if importances[indices[f]] > 0.0001:
            print(
                str(f + 1), str(indices[f]), feature_name[indices[f]], str(importances[indices[f]])
            )


def split_train(df1, ntrain=10000, ntest=100000, colused=None):
    n1 = len(df1[df1["y"] == 0])
    dft = pd.concat(
        (
            df1[df1["y"] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1["y"] == 1) & (df1["def"] > 201803)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft["y"].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1["y"] == 0])
    dft2 = pd.concat(
        (
            df1[df1["y"] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1["y"] == 1) & (df1["def"] > 201703) & (df1["def"] < 201804)].iloc[:, :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2["y"].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test


def split_train2(df1, ntrain=10000, ntest=100000, colused=None, nratio=0.4):
    n1 = len(df1[df1["y"] == 0])
    n2 = len(df1[df1["y"] == 1])
    n2s = int(n2 * nratio)  # 80% of default

    #### Test data
    dft = pd.concat(
        (
            df1[df1["y"] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1["y"] == 1)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft["y"].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1["y"] == 0])
    dft2 = pd.concat(
        (
            df1[df1["y"] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1["y"] == 1)].iloc[np.random.choice(n2, n2s, False), :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2["y"].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test


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
    import catboost

    pa = dict2(pars)

    if Xcolname is None:
        Xcolname = [str(i) for i in range(0, Xtrain.shape[1])]
    train_df = pd.DataFrame(Xtrain, Xcolname)
    cat_features_ids = Xcolname

    clf = catboost.CatBoostClassifier(
        learning_rate=pa.learning_rate,
        iterations=pa.iterations,
        random_seed=pa.random_seed,
        loss_function=pa.loss_function,
    )
    clf.fit(Xtrain, Ytrain, cat_features=cat_features_ids)

    Y_pred = clf.predict(Xtrain)

    cm = sk.metrics.confusion_matrix(Ytrain, Y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return clf, cm, cm_norm


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
    from tpot import TPOTClassifier, TPOTRegressor

    X_train, X_test, y_train, y_test = train_test_split(Xmat, y, train_size=0.5, test_size=0.5)

    if model_type == "regressor":
        tpot = TPOTRegressor(
            generations=generation, population_size=population_size, verbosity=verbosity
        )
    elif model_type == "classifier":
        tpot = TPOTClassifier(
            generations=generation, population_size=population_size, verbosity=verbosity
        )

    tpot.fit(X_train, y_train)
    print((tpot.score(X_test, y_test)))
    file1 = (
        DIRCWD
        + "/"
        + outfolder
        + "/tpot_regression_pipeline_"
        + str(np.random.randint(1000, 9999))
        + ".py"
    )
    tpot.export(file1)
    return file1


def sk_params_search_best(
    Xmat,
    Ytarget,
    model1,
    param_grid={"alpha": np.linspace(0, 1, 5)},
    method="gridsearch",
    param_search={"scoretype": "r2", "cv": 5, "population_size": 5, "generations_number": 3},
):
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
    p = param_search

    from sklearn.metrics import make_scorer, r2_score

    if param_search["scoretype"] == "r2":
        myscore = make_scorer(r2_score, sample_weight=None)

    if method == "gridsearch":
        from sklearn.grid_search import GridSearchCV

        grid = GridSearchCV(
            model1, param_grid, cv=p["cv"], scoring=myscore
        )  # 20-fold cross-validation
        grid.fit(Xmat, Ytarget)
        return grid.best_score_, grid.best_params_

    if method == "genetic":
        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        from sklearn.model_selection import StratifiedKFold

        # paramgrid = {"alpha":  np.linspace(0,1, 20) , "l1_ratio": np.linspace(0,1, 20) }
        cv = EvolutionaryAlgorithmSearchCV(
            estimator=model1,
            params=param_grid,
            scoring=p["scoretype"],
            cv=StratifiedKFold(y, n_folds=p["cv"]),
            verbose=True,
            population_size=p["population_size"],
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=p["generations_number"],
        )

        cv.fit(Xmat, Ytarget)
        return cv.best_score_, cv.best_params_
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
    from sklearn.grid_search import GridSearchCV

    grid = GridSearchCV(
        kde, {"bandwidth": np.linspace(0.1, 1.0, 30)}, cv=20
    )  # 20-fold cross-validation
    grid.fit(x[:, None])
    return grid.best_params_


def sk_distribution_kernel_sample(kde=None, n=1):
    """
  kde = sm.nonparametric.KDEUnivariate(np.array(Y[Y_cluster==0],dtype=np.float64))
  kde = sm.nonparametric.KDEMultivariate()  # ... you already did this
 """

    from scipy.optimize import brentq

    samples = np.zeros(n)

    # 1-d root-finding  F-1(U) --> Sample
    def func(x):
        return kde.cdf([x]) - u

    for i in range(0, n):
        u = np.random.random()  # sample
        samples[i] = brentq(func, -999, 999)  # read brentq-docs about these constants
    return samples


def sk_correl_rank(correl=[[1, 0], [0, 1]]):
    """ Correl Ranking:  Col i, Col j, Correl_i_j, Abs_Correl_i_j    """
    m, n = np.shape(correl)
    correl_rank = np.zeros((n * (n - 1) / 2, 3), dtype=np.float32)
    k = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            k += 1
            correl_rank[k, 0] = i
            correl_rank[k, 1] = j
            correl_rank[k, 2] = correl[i, j]
            correl_rank[k, 3] = abs(correl[i, j])
    correl_rank = util.sortcol(correl_rank, 3, asc=False)
    return correl_rank


def sk_error_r2(Ypred, y_true, sample_weight=None, multioutput=None):
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, Ypred, sample_weight=sample_weight, multioutput=multioutput)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    if r <= -1:
        return -1
    else:
        return r


def sk_error_rmse(Ypred, Ytrue):
    aux = np.sqrt(np.sum((Ypred - Ytrue) ** 2)) / len(Ytrue)
    return "Error:", aux, "Error/Stdev:", aux / np.std(Ytrue)


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
    if metric == "jaccard":
        return fast.distance_jaccard_X(Xmat)

    else:  # if metric=='euclidian'
        return sci.spatial.distance.squareform(
            sci.spatial.distance.pdist(Xmat, metric=metric, p=2, w=None, V=None, VI=None)
        )


def sk_cluster(
    Xmat,
    method="kmode",
    args=(),
    kwds={"metric": "euclidean", "min_cluster_size": 150, "min_samples": 3},
    isprint=1,
    preprocess={"norm": False},
):
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
    if method == "kmode":
        # Kmode clustering data nbCategory,  NbSample, NbFeatures
        km = kmodes.kmodes.KModes(*args, **kwds)
        Xclus_class = km.fit_predict(Xmat)
        return Xclus_class, km, km.cluster_centroids_  # Class, km, centroid

    if method == "hdbscan":
        import hdbscan

        Xcluster_id = hdbscan.HDBSCAN(*args, **kwds).fit_predict(Xmat)
        print(("Nb Cluster", len(np.unique(Xcluster_id))))
        return Xcluster_id

    if method == "kmeans":
        from sklearn.cluster import KMeans

        if preprocess["norm"]:
            stdev = np.std(Xmat, axis=0)
            Xmat = (Xmat - np.mean(Xmat, axis=0)) / stdev

        sh = Xmat.shape
        Xdim = 1 if len(sh) < 2 else sh[1]  # 1Dim vector or 2dim-3dim vector
        print((len(Xmat.shape), Xdim))
        if Xdim == 1:
            Xmat = Xmat.reshape((sh[0], 1))

        kmeans = KMeans(**kwds)  #  KMeans(n_clusters= nbcluster)
        kmeans.fit(Xmat)
        centroids, labels = kmeans.cluster_centers_, kmeans.labels_

        if isprint:
            import matplotlib.pyplot as plt

            colors = ["g.", "r.", "y.", "b.", "k."]
            if Xdim == 1:
                for i in range(0, sh[0], 5):
                    plt.plot(Xmat[i], colors[labels[i]], markersize=5)
                plt.show()
            elif Xdim == 2:
                for i in range(0, sh[0], 5):
                    plt.plot(Xmat[i, 0], Xmat[i, 1], colors[labels[i]], markersize=2)
                plt.show()
            else:
                print("Cannot Show higher than 2dim")

        return labels, centroids


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
    import copy

    if isreset == 2:
        print("Traditionnal Optim, no saving")
        res = sci.optimize.differential_evolution(obj_fun, bounds=bounds, maxiter=maxiter)
        xbest, fbest, solver, i = res.x, res.fun, "", maxiter
    else:  # iterative solver
        print("Iterative Solver ")
        if name1 != "":  # wtih file
            print("/batch/" + name1)
            solver2 = load_obj("/batch/" + name1)
            imin = int(name1[-3:]) + 1
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            solver.population = copy.deepcopy(solver2.population)
            solver.population_energies = copy.deepcopy(solver2.population_energies)
            del solver2

        elif solver1 is not None:  # Start from zero
            solver = copy.deepcopy(solver1)
            imin = 0
        else:
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            imin = 0

        name1 = "/batch/solver_" + name1
        fbest0 = 1500000.0
        for i in range(imin, imin + maxiter):
            xbest, fbest = next(solver)
            print(0, i, fbest, xbest)
            res = (copy.deepcopy(solver), i, xbest, fbest)
            try:
                util.save_obj(solver, name1 + util.date_now() + "_" + util.np_int_tostr(i))
                print((name1 + util.date_now() + "_" + util.np_int_tostr(i)))
            except:
                pass
            if np.mod(i + 1, 11) == 0:
                if np.abs(fbest - fbest0) < 0.001:
                    break
                fbest0 = fbest

    return fbest, xbest, solver


######## Valuation model template  ##########################################################
class sk_model_template1(sk.base.BaseEstimator):
    def __init__(self, alpha=0.5, low_y_cut=-0.09, high_y_cut=0.09, ww0=0.95):
        from sklearn.linear_model import Ridge

        self.alpha = alpha
        self.low_y_cut, self.high_y_cut, self.ww0 = 1000.0 * low_y_cut, 1000.0 * high_y_cut, ww0
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, Y=None):
        X, Y = X * 100.0, Y * 1000.0

        y_is_above_cut = Y > self.high_y_cut
        y_is_below_cut = Y < self.low_y_cut
        y_is_within_cut = ~y_is_above_cut & ~y_is_below_cut
        if len(y_is_within_cut.shape) > 1:
            y_is_within_cut = y_is_within_cut[:, 0]

        self.model.fit(X[y_is_within_cut, :], Y[y_is_within_cut])

        r2 = self.model.score(X[y_is_within_cut, :], Y[y_is_within_cut])
        print(("R2:", r2))
        print(("Inter", self.model.intercept_))
        print(("Coef", self.model.coef_))

        self.ymedian = np.median(Y)
        return self, r2, self.model.coef_

    def predict(self, X, y=None, ymedian=None):
        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian

        Y = Y / 1000.0
        return Y

    def score(self, X, Ytrue=None, ymedian=None):
        from sklearn.metrics import r2_score

        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian
        Y = Y / 1000.0
        return r2_score(Ytrue, Y)


############################################################################
# ---------------------             ----------------
"""
 Reshape your data either using X.reshape(-1, 1) if your data has a single feature or
  X.reshape(1, -1) if it contains a single sample.

"""


def sk_feature_importance(clfrf, feature_name):
    importances = clfrf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(0, len(feature_name)):
        if importances[indices[f]] > 0.0001:
            print(
                str(f + 1), str(indices[f]), feature_name[indices[f]], str(importances[indices[f]])
            )


# -------- SK Learn TREE UTIL----------------------------------------------------------------
def sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1=1, njobs=1):
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
    clfrf = sk.ensemble.RandomForestClassifier(
        n_estimators=nbtree,
        max_depth=maxdepth,
        max_features="sqrt",
        criterion="entropy",
        n_jobs=njobs,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    clfrf.fit(Xtrain, Ytrain)
    Y_pred = clfrf.predict(Xtrain)

    cm = sk.metrics.confusion_matrix(Ytrain, Y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint1:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return clfrf, cm, cm_norm


def sk_gen_ensemble_weight(vv, acclevel, maxlevel=0.88):
    imax = min(acclevel, len(vv))
    estlist = np.empty(imax, dtype=np.object)
    estww = []
    for i in range(0, imax):
        # if vv[i,3]> acclevel:
        estlist[i] = vv[i, 1]
        estww.append(vv[i, 3])
        # print 5
    # Log Proba Weighted + Impact of recent False discovery
    estww = np.log(1 / (maxlevel - np.array(estww) / 2.0))
    # estww= estww/np.sum(estww)
    # return np.array(estlist), np.array(estww)
    return estlist, np.array(estww)


def sk_votingpredict(estimators, voting, ww, X_test):
    ww = ww / np.sum(ww)
    Yproba0 = np.zeros((len(X_test), 2))
    Y1 = np.zeros((len(X_test)))

    for k, clf in enumerate(estimators):
        Yproba = clf.predict_proba(X_test)
        Yproba0 = Yproba0 + ww[k] * Yproba

    for k in range(0, len(X_test)):
        if Yproba0[k, 0] > Yproba0[k, 1]:
            Y1[k] = -1
        else:
            Y1[k] = 1
    return Y1, Yproba0


def sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base=" "):
    """Produce psuedo-code for decision tree.
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (output) names.
    spacer_base -- used for spacing code (default: "    ").
    """
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if threshold[node] != -2:
            print((spacer + "if " + features[node] + " <= " + str(threshold[node]) + " :"))
            #            print(spacer + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) :")
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth + 1)
            print(("" + spacer + "else :"))
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], depth + 1)
        #     print(spacer + "")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(
                    (
                        spacer
                        + "return "
                        + str(target_name)
                        + " ( "
                        + str(target_count)
                        + ' examples )"'
                    )
                )

    recurse(left, right, threshold, features, 0, 0)


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
        self.lrule = np.empty((3, 20), dtype=np.object)
        self.nrule = 20
        sh = np.shape(state)
        self.tmax = sh[0]
        self.nstate = sh[1]
        sh = np.shape(trigger)
        self.ktrigger = sh[0]
        self.ntrigger = sh[1]

        if len(colname) > 1:
            self.colname = colname
        else:
            self.colname = ["a" + str(i) for i in range(0, self.nstate)]

        self.state = util.np_torecarray(state, self.colname)
        self.trigger = util.np_torecarray(trigger, self.colname)

    def addrule(self, rulefun, name="", desc=""):
        kid = util.findnone(self.lrule[0, :])
        kid2 = util.find(name, self.lrule[1, :])
        if kid2 != -1 and name != "":
            print("Name already exist !")
        else:
            if kid == -1:
                lrule = util.np_addcolumn(self.lrule, 50)
                kid = self.nrule

            try:
                test = rulefun(self.state, self.trigger, 1)
                self.lrule[0, kid] = copy.deepcopy(rulefun)
                self.lrule[1, kid] = name
                self.lrule[2, kid] = desc
            except ValueError as e:
                print(("Error with the function" + str(e)))

    def eval(self, idrule, t, ktrig=0):
        if isinstance(idrule, str):  # Evaluate by name
            kid = util.find(idrule, self.lrule[1, :])
            if kid != -1:
                return self.lrule[0, kid](self.state, self.trigger, t)
            else:
                print(("cannot find " + idrule))
        else:
            return self.lrule[0, idrule](self.state, self.trigger, t)

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

#### ML metrics


def sk_showmetrics(y_test, ytest_pred, ytest_proba, target_names=["0", "1"]):
    #### Confusion matrix
    mtest = sk_showconfusion(y_test, ytest_pred, isprint=False)
    # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
    auc = roc_auc_score(y_test, ytest_proba)  #
    gini = 2 * auc - 1
    acc = accuracy_score(y_test, ytest_pred)
    f1macro = sk.metrics.f1_score(y_test, ytest_pred, average="macro")

    print("Test confusion matrix")
    print(mtest[0])
    print(mtest[1])
    print("auc " + str(auc))
    print("gini " + str(gini))
    print("acc " + str(acc))
    print("f1macro " + str(f1macro))
    print("Nsample " + str(len(y_test)))

    print(classification_report(y_test, ytest_pred, target_names=target_names))

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.show()


def sk_showmetrics2(y_test, ytest_pred, ytest_proba, target_names=["0", "1"]):
    #### Confusion matrix
    # mtest  = sk_showconfusion( y_test, ytest_pred, isprint=False)
    # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
    auc = roc_auc_score(y_test, ytest_proba)  #
    gini = 2 * auc - 1
    acc = accuracy_score(y_test, ytest_pred)
    return auc, gini, acc


def clf_prediction_score(clf, df1, cols, outype="score"):
    def score_calc(yproba, pnorm=1000.0):
        yy = np.log(0.00001 + (1 - yproba) / (yproba + 0.001))
        # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
        # return  np.maximum( 0.01 , yy )    ## Error it bias proba
        return yy

    X_all = df1[cols].values

    yall_proba = clf.predict_proba(X_all)[:, 1]
    yall_pred = clf.predict(X_all)
    try:
        y_all = df1["y"].values
        sk_showmetrics(y_all, yall_pred, yall_proba)
    except:
        pass

    yall_score = score_calc(yall_proba)
    yall_score = (
        1000 * (yall_score - np.min(yall_score)) / (np.max(yall_score) - np.min(yall_score))
    )

    if outype == "score":
        return yall_score
    if outype == "proba":
        return yall_proba, yall_pred


def sk_showconfusion(clfrf, X_train, Y_train, isprint=True):
    Y_pred = clfrf.predict(X_train)
    cm = sk.metrics.confusion_matrix(Y_train, Y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return cm, cm_norm, cm_norm[0, 0] + cm_norm[1, 1]
