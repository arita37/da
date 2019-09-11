# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas



"""
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci


########### LOCAL ##################################################################################
print("os.getcwd", os.getcwd())


def ztest():
    import sklearn as sk
    print(sk)



def pd_num_segment_limit(
    df, col_score="scoress", coldefault="y", ntotal_default=491, def_list=None, nblock=20.0
):
    """
    Calculate Segmentation of colum using rule based.
    :param df:
    :param col_score:
    :param coldefault:
    :param ntotal_default:
    :param def_list:
    :param nblock:
    :return:
    """

    if def_list is None:
        def_list = np.ones(21) * ntotal_default / nblock

    df["scoress_bin"] = df[col_score].apply(lambda x: np.floor(x / 1.0) * 1.0)
    dfs5 = (
        df.groupby("scoress_bin")
        .agg({col_score: "mean", coldefault: {"sum", "count"}})
        .reset_index()
    )
    dfs5.columns = [x[0] if x[0] == x[1] else x[0] + "_" + x[1] for x in dfs5.columns]
    dfs5 = dfs5.sort_values(col_score, ascending=False)
    # return dfs5

    l2 = []
    k = 1
    ndef, nuser = 0, 0
    for i, x in dfs5.iterrows():
        if k > nblock:
            break
        nuser = nuser + x[coldefault + "_count"]
        ndef = ndef + x[coldefault + "_sum"]
        pdi = ndef / nuser

        if ndef > def_list[k - 1]:
            # if  pdi > pdlist[k] :
            l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
            k = k + 1
            ndef, nuser = 0, 0
        l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
    l2 = pd.DataFrame(l2, columns=[col_score, "kaiso3", "pd", "ndef", "nuser"])
    return l2


def fun_get_segmentlimit(x, l1):
    """
    ##### Get Kaiso limit ###############################################################
    :param x:
    :param l1:
    :return :
    """
    for i in range(0, len(l1)):
        if x >= l1[i]:
            return i + 1
    return i + 1







