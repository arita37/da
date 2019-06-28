## -*- coding: utf-8 -*-
#"""    Data Analysis Utilities   """
## type of data
#"""
#df : dataframe
#Xmat :  Numpy Matrix values
#Ytarget : Value to be predicted, Class
#col: column,    row: row
#
#"""
#import copy
#import itertools
#import math
#import os
#import re
#import sys
#from calendar import isleap
#from collections import OrderedDict
#from datetime import datetime, timedelta
#
#import arrow
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import pylab as pl
#import requests
#import scipy as sci
#import sklearn as sk
#import statsmodels as sm
#from dateutil.parser import parse
#from sklearn import covariance, linear_model, model_selection
#from sklearn.cluster import dbscan, k_means
#from sklearn.decomposition import PCA, pca
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
#                              GradientBoostingClassifier, RandomForestClassifier)
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#
#import datanalysis as da
#import kmodes
#import util
#from attrdict import AttrDict as dict2
#from kmodes.kmodes import KModes
#from tabulate import tabulate
#
##########################################################################################################
## DIRCWD= os.environ["DIRCWD"]; os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
## import configmy; CFG, DIRCWD= configmy.get(config_file="_ROOT", output= ["_CFG", "DIRCWD"])
#DIRCWD = "./"
## os.chdir(DIRCWD)
## sys.path.append(DIRCWD + "/aapackage")
#
#
## __path__ = DIRCWD + "/aapackage/"
#__version__ = "1.0.0"
#__file__ = "datanalysis.py"
#
#
#
############## Pandas Processing   ######################################################################
#
#
#
#############################################################################
#
#
####################################################################################################################
############################# UNIT TEST ############################################################################
#if __name__ == "__main__":
#    import argparse
#
#    ppa = argparse.ArgumentParser()  # Command Line input
#    ppa.add_argument("--do", type=str, default="action", help="test / test02")
#    arg = ppa.parse_args()
#
#
#if __name__ == "__main__" and arg.do == "test":
#    print(__file__)
#    try:
#        import util
#
#        UNIQUE_ID = util.py_log_write(DIRCWD + "/aapackage/ztest_log_all.txt", "datanalysis")
#
#        ##########################################################################################################
#        import numpy as np, pandas as pd, scipy as sci
#        import datanalysis as da
#
#        print(da)
#
#        vv = np.random.rand(1, 10)
#        mm = np.random.rand(100, 5)
#        df = pd.DataFrame(mm, columns=["a", "b", "c", "d", "e"])
#
#        Xtrain = mm
#        Ytrain = np.random.randint(0, 1, len(Xtrain))
#        clfrf = da.sk_tree(Xtrain=Xtrain, Ytrain=Ytrain, nbtree=2, maxdepth=5, isprint1=0)
#        print(clfrf)
#
#        da.sk_cluster(Xmat=Xtrain, method="kmeans", kwds={"n_clusters": 5})
#
#        ##########################################################################################################
#        print(
#            "\n\n"
#            + UNIQUE_ID
#            + " ###################### End:"
#            + arrow.utcnow().to("Japan").format()
#            + "###########################"
#        )
#        sys.stdout.flush()
#    except Exception as e:
#        util.py_exception_print()
#
#
#"""
#  try :
#
#  except Exception as e: print(e)
#
#"""
