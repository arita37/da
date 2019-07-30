# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""
import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sci
from dateutil.parser import parse

import sklearn as sk
from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

####################################################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())


####################################################################################################
class dict2(object):
    ## Dict with attributes
    def __init__(self, d):
        self.__dict__ = d


def np_transform_pca(X, dimpca=2, whiten=True):
    """Project ndim data into dimpca sub-space  """
    pca = PCA(n_components=dimpca, whiten=whiten).fit(X)
    return pca.transform(X)


def sk_distribution_kernel_bestbandwidth(X, kde):
    """Find best Bandwidht for a  given kernel
  :param kde:
  :return:
 """
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(
        kde, {"bandwidth": np.linspace(0.1, 1.0, 30)}, cv=20
    )  # 20-fold cross-validation
    grid.fit(X[:, None])
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
