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
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
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







def stat_hypothesis_test_permutation(df, variable, classes, repetitions):

    """Test whether two numerical samples
    come from the same underlying distribution,
    using the absolute difference between the means.
    table: name of table containing the sample
    variable: label of column containing the numerical variable
    classes: label of column containing names of the two samples
    repetitions: number of random permutations"""

    t = df[[ variable, classes]]

    # Find the observed test statistic
    means_table = t.groupby(classes).agg(np.mean)
    obs_stat = abs(means_table.column(1).item(0) - means_table.column(1).item(1))

    # Assuming the null is true, randomly permute the variable
    # and collect all the generated test statistics
    stats = make_array()
    for i in np.arange(repetitions):
        shuffled_var = t.select(variable).sample(with_replacement=False).column(0)
        shuffled = t.select(classes).with_column('Shuffled Variable', shuffled_var)
        m_tbl = shuffled.group(classes, np.mean)
        new_stat = abs(m_tbl.column(1).item(0) - m_tbl.column(1).item(1))
        stats = np.append(stats, new_stat)

    # Find the empirical P-value:
    emp_p = np.count_nonzero(stats >= obs_stat)/repetitions

    # Draw the empirical histogram of the tvd's generated under the null,
    # and compare with the value observed in the original sample
    Table().with_column('Test Statistic', stats).hist(bins=20)
    plots.title('Empirical Distribution Under the Null')
    print('Observed statistic:', obs_stat)
    print('Empirical P-value:', emp_p)
    
    
    








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
