# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy


https://stats.stackexchange.com/questions/222558/classification-evaluation-metrics-for-highly-imbalanced-data




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
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD, pca)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
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
        "/" + outfolder + "/tpot_regression_pipeline_" + str(np.random.randint(1000, 9999)) + ".py"
    )
    tpot.export(file1)
    return file1



