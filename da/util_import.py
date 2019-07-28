# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""
import os
import copy
from collections import OrderedDict
import dateutil


import numpy as np
import pandas as pd

import scipy as sci
import sklearn as sk


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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, make_scorer


# from attrdict import AttrDict as dict2
# from kmodes.kmodes import KModes
# from tabulate import tabulate


try:
    from catboost import CatBoostClassifier, Pool, cv
except Exception as e:
    print(e)


####################################################################################################
print( os.getcwd() )



####################################################################################################
class dict2(object):
    def __init__(self, d):
        self.__dict__ = d





