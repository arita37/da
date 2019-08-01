# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""
import copy
import os
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pandas as pd
import scipy as sci
from dateutil.parser import parse

import sklearn as sk
from matplotlib import pyplot as plt
from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def import_(abs_module_path, class_name=None):
    try:
        module_object = import_module(abs_module_path)
        print("imported", module_object)
        if class_name is None:
            return module_object
        target_class = getattr(module_object, class_name)
        return target_class
    except Exception as e:
        print(abs_module_path, class_name, e)


# from attrdict import AttrDict as dict2
# from kmodes.kmodes import KModes
# from tabulate import tabulate


########### Dynamic Import   #######################################################################
# EvolutionaryAlgorithmSearchCV = import_("evolutionary_search", "EvolutionaryAlgorithmSearchCV")
esearch = import_("evolutionary_search")
lgb = import_("lightgbm")
kmodes = import_("kmodes")
catboost = import_("catboost")
tpot = import_("tpot")

####################################################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())


class dict2(object):
    def __init__(self, d):
        self.__dict__ = d


####################################################################################################
def np_transform_pca(X, dimpca=2, whiten=True):
    """Project ndim data into dimpca sub-space  """
    pca = PCA(n_components=dimpca, whiten=whiten).fit(X)
    return pca.transform(X)





def split_train_test(X, y, split_ratio=0.8):
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=split_ratio, random_state=42, shuffle=False
    )
    print("train_X shape:", train_X.shape)
    print("val_X shape:", val_X.shape)

    print("train_y shape:", train_y.shape)
    print("val_y shape:", val_y.shape)

    return train_X, val_X, train_y, val_y


def split_train(df1, ntrain=10000, ntest=100000, colused=None, coltarget=None):
    n1 = len(df1[df1[coltarget] == 0])
    dft = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1[coltarget] == 1) & (df1["def"] > 201803)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft[coltarget].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1[coltarget] == 0])
    dft2 = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1[coltarget] == 1) & (df1["def"] > 201703) & (df1["def"] < 201804)].iloc[:, :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2[coltarget].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test


def split_train2(df1, ntrain=10000, ntest=100000, colused=None, coltarget=None, nratio=0.4):
    n1 = len(df1[df1[coltarget] == 0])
    n2 = len(df1[df1[coltarget] == 1])
    n2s = int(n2 * nratio)  # 80% of default

    #### Test data
    dft = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1[coltarget] == 1)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft[coltarget].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1[coltarget] == 0])
    dft2 = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1[coltarget] == 1)].iloc[np.random.choice(n2, n2s, False), :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2[coltarget].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test


def model_lightgbm_kfold(
    # LightGBM GBDT with KFold or Stratified KFold
    df,
    colname=None,
    num_folds=2,
    stratified=False,
    colexclude=None,
    debug=False,
):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df.shape[0])
    feature_importance_df = pd.DataFrame()
    # colname = [f for f in df.columns if f not in colexclude]

    regs = []
    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[colname], df["is_match"])):
        train_x, train_y = df[colname].iloc[train_idx], df["is_match"].iloc[train_idx]
        valid_x, valid_y = df[colname].iloc[valid_idx], df["is_match"].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x, label=valid_y, free_raw_data=False)

        # params optimized by optuna
        params = {
            "max_depth": -1,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 2 ** 12 - 1,
            "colsample_bytree": 0.28,
            "objective": "binary",
            "n_jobs": -1,
        }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=["train", "test"],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=100,
        )
        regs.append(reg)

    return regs


def model_catboost_classifier(
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
    X_train, X_test, y_train, y_test = train_test_split(Xmat, y, train_size=0.5, test_size=0.5)

    if model_type == "regressor":
        clf = tpot.TPOTRegressor(
            generations=generation, population_size=population_size, verbosity=verbosity
        )
    elif model_type == "classifier":
        clf = tpot.TPOTClassifier(
            generations=generation, population_size=population_size, verbosity=verbosity
        )

    clf.fit(X_train, y_train)
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


def sk_score_get(name="r2"):
    from sklearn.metrics import make_scorer, r2_score, roc_auc_score, mean_squared_error

    if name == "r2":
        return sk.metrics.make_scorer(r2_score, sample_weight=None)

    if name == "auc":
        return sk.metrics.make_scorer(r2_score, sample_weight=None)


def sk_params_search_best(
    clf,
    X,
    y,
    param_grid={"alpha": np.linspace(0, 1, 5)},
    method="gridsearch",
    param_search={"scorename": "r2", "cv": 5, "population_size": 5, "generations_number": 3},
):
    """
   Genetic: population_size=5, ngene_mutation_prob=0.10,,gene_crossover_prob=0.5, tournament_size=3,  generations_number=3

    :param X:
    :param y:
    :param clf:
    :param param_grid:
    :param method:
    :param param_search:
    :return:
  """
    p = param_search
    myscore = sk_score_get(p["scorename"])

    if method == "gridsearch":
        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(clf, param_grid, cv=p["cv"], scoring=myscore)
        grid.fit(X, y)
        return grid.best_score_, grid.best_params_

    if method == "genetic":
        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        from sklearn.model_selection import StratifiedKFold

        # paramgrid = {"alpha":  np.linspace(0,1, 20) , "l1_ratio": np.linspace(0,1, 20) }
        cv = EvolutionaryAlgorithmSearchCV(
            estimator=clf,
            params=param_grid,
            scoring=myscore,
            cv=StratifiedKFold(y),
            verbose=True,
            population_size=p["population_size"],
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=p["generations_number"],
        )

        cv.fit(X, y)
        return cv.best_score_, cv.best_params_


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
    print("Error:", aux, "Error/Stdev:", aux / np.std(Ytrue))
    return aux / np.std(Ytrue)


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

        kmeans = KMeans(**kwds)  # KMeans(n_clusters= nbcluster)
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


######## Valuation model template  ##########################################################
class model_template1(sk.base.BaseEstimator):
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


def sk_model_ensemble_weight(vv, acclevel, maxlevel=0.88):
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


#### ML metrics


def sk_showconfusion(Y, Ypred, isprint=True):
    cm = sk.metrics.confusion_matrix(Y, Ypred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return cm, cm_norm, cm_norm[0, 0] + cm_norm[1, 1]


def sk_showmetrics(y_test, ytest_pred, ytest_proba, target_names=["0", "1"], return_stat=0):
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

    # Show roc curve
    try:
        fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr, marker=".")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.show()
    except Exception as e:
        print(e)

    if return_stat:
        return {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}


def model_logistic_score(clf, df1, cols, coltarget, outype="score"):
    """

    :param clf:
    :param df1:
    :param cols:
    :param outype:
    :return:
    """

    def score_calc(yproba, pnorm=1000.0):
        yy = np.log(0.00001 + (1 - yproba) / (yproba + 0.001))
        # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
        # return  np.maximum( 0.01 , yy )    ## Error it bias proba
        return yy

    X_all = df1[cols].values

    yall_proba = clf.predict_proba(X_all)[:, 1]
    yall_pred = clf.predict(X_all)
    try:
        y_all = df1[coltarget].values
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


def sk_model_eval_regression(clf, istrain=1, Xtrain=None, ytrain=None, Xval=None, yval=None):
    if istrain:
        clf.fit(Xtrain, ytrain)

    CV_score = -cross_val_score(clf, Xtrain, ytrain, scoring="neg_mean_absolute_error", cv=4)

    print("CV score: ", CV_score)
    print("CV mean: ", CV_score.mean())
    print("CV std:", CV_score.std())

    train_y_predicted_logReg = clf.predict(Xtrain)
    val_y_predicted_logReg = clf.predict(Xval)

    print("\n")
    print("Score on logReg training set:", mean_absolute_error(ytrain, train_y_predicted_logReg))
    print("Score on logReg validation set:", mean_absolute_error(yval, val_y_predicted_logReg))

    return clf, train_y_predicted_logReg, val_y_predicted_logReg


def sk_model_eval_classification(clf, istrain=1, Xtrain=None, ytrain=None, Xtest=None, ytest=None):
    if istrain:
        print("############# Train dataset  ####################################")
        clf.fit(Xtrain, ytrain)
        ytrain_proba = clf.predict_proba(Xtrain)[:, 1]
        ytrain_pred = clf.predict(Xtrain)
        sk_showmetrics(ytrain, ytrain_pred, ytrain_proba)

    print("############# Test dataset  #########################################")
    ytest_proba = clf.predict_proba(Xtest)[:, 1]
    ytest_pred = clf.predict(Xtest)
    sk_showmetrics(ytest, ytest_pred, ytest_proba)

    return clf, {"ytest_pred": ytest_pred}


###################################################################################################
def sk_feature_impt(clf, colname, model_type="logistic"):
    """
       Feature importance with colname
    :param clf:  model or colnum with weights
    :param colname:
    :return:
    """
    if model_type == "logistic" :
       dfeatures = pd.DataFrame(
          {"feature":    colname, "weight": clf.coef_[0], 
           "weight_abs": np.abs(clf.coef_[0])}
       ).sort_values("weight_abs", ascending=False)
       dfeatures["rank"] = np.arange(0, len(dfeatures))
       return dfeatures
   
    else:
      # RF, Xgboost, LightGBM
      if isinstance(clf, list) or isinstance(clf, (np.ndarray, np.generic) ) :
         importances = clf 
      else :
         importances = clf.feature_importances_
      rank = np.argsort(importances)[::-1]
      d = {"col": [], "rank": [], "weight": []}
      for i in range(0, len(colname)):
        d["rank"].append(rank[i])
        d["col"].append(colname[rank[i]])
        d["weight"].append(importances[rank[i]])

      return pd.DataFrame(d)



    

def sk_feature_selection(clf, method="f_classif", colname=None, kbest=50, Xtrain=None, ytrain=None):
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression

    if method == "f_classif":
        clf_best = SelectKBest(f_classif, k=kbest).fit(Xtrain, ytrain)

    if method == "f_regression":
        clf_best = SelectKBest(f_regression, k=kbest).fit(Xtrain, ytrain)

    mask = clf_best.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, colname):
        if bool:
            new_features.append(feature)

    return new_features


def sk_feature_evaluation(clf, df, kbest=30, colname_best=None, dfy=None):
    clf2 = copy.deepcopy(clf)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        df[colname_best].values, dfy.values, random_state=42, test_size=0.5, shuffle=True
    )
    print(Xtrain.shape, ytrain.shape)

    df = {x: [] for x in ["col", "auc", "acc", "f1macro", "confusion"]}
    for i in range(1, len(colname_best)):
        print("########## ", colname_best[:i])
        if i > kbest:
            break
        clf.fit(Xtrain[:, :i], ytrain)
        ytest_proba = clf.predict_proba(Xtest[:, :i])[:, 1]
        ytest_pred = clf.predict(Xtest[:, :i])
        s = sk_showmetrics(ytest, ytest_pred, ytest_proba, return_stat=1)

        # {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}

        df["col"].append(str(colname_best[:i]))
        df["auc"].append(s["auc"])
        df["acc"].append(s["acc"])
        df["f1macro"].append(s["f1macro"])
        df["confusion"].append(s["confusion"])

    df = pd.DataFrame(df)
    return df


def sk_feature_drift_covariance(dftrain, dftest, colname, nsample=10000):
    n1 = nsample if len(dftrain) > nsample else len(dftrain)
    n2 = nsample if len(dftest) > nsample else len(dftest)
    train = dftrain[colname].sample(n1, random_state=12)
    test = dftest[colname].sample(n2, random_state=11)

    ## creating a new feature origin
    train["origin"] = 0
    test["origin"] = 1

    ## combining random samples
    combi = train.append(test)
    y = combi["origin"]
    combi.drop("origin", axis=1, inplace=True)

    ## modelling
    model = RandomForestClassifier(n_estimators=50, max_depth=7, min_samples_leaf=5)
    drop_list = []
    for i in combi.columns:
        score = cross_val_score(model, pd.DataFrame(combi[i]), y, cv=2, scoring="roc_auc")

        if np.mean(score) > 0.8:
            drop_list.append(i)
        print(i, np.mean(score))
    return drop_list


def sk_model_eval_classification_cv(clf, X, y, test_size=0.5, ncv=1, method="random"):
    """
    :param clf:
    :param X:
    :param y:
    :param test_size:
    :param ncv:
    :param method:
    :return:
    """
    if method == "kfold":
        kf = StratifiedKFold(n_splits=ncv, shuffle=True)
        clf_list = {}
        for i, itrain, itest in enumerate(kf.split(X, y)):
            print("###")
            Xtrain, Xtest = X[itrain], X[itest]
            ytrain, ytest = y[itrain], y[itest]
            clf_list[i], _ = sk_model_eval_classification(clf, 1, Xtrain, ytrain, Xtest, ytest)

    else:
        clf_list = {}
        for i in range(0, ncv):
            print("############# CV-{i}######################################".format(i=i))
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, shuffle=True)

            clf_list[i], _ = sk_model_eval_classification(clf, 1, Xtrain, ytrain, Xtest, ytest)

    return clf_list


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
def sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval=1):
    pass
Plot the cLuster using specific Algo
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

"""


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
    
    
    
https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api

X_train, X_test, up_train, up_test, r_train, r_test, u_train, u_test, d_train, d_test = model_selection.train_test_split(
    X, up, r, universe, d, test_size=0.25, random_state=99)


train_cols = X_train.columns.tolist()

train_data = lgb.Dataset(X_train, label=up_train.astype(int), 
                         feature_name=train_cols)
test_data = lgb.Dataset(X_test, label=up_test.astype(int), 
                        feature_name=train_cols, reference=train_data)
                        
                        
# LGB parameters:
params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary',
          'num_leaves': 2000,
          'min_data_in_leaf': 200,
          'max_bin': 200,
          'max_depth': 16,
          'seed': 2018,
          'nthread': 10,}


# LGB training:
lgb_model = lgb.train(params, train_data, 
                      num_boost_round=1000, 
                      valid_sets=(test_data,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
                      
                      

# DF, based on which importance is checked
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_importance)


# Plot summary_plot
shap.summary_plot(shap_values, X_importance)
                      

# Plot summary_plot as barplot:
shap.summary_plot(shap_values, X_importance, plot_type='bar')


shap.dependence_plot("returnsClosePrevRaw10_lag_3_mean", shap_values, X_importance)



"""
