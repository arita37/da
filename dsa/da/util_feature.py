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
from sklearn.cluster import KMeans



########### LOCAL ##################################################################################
print("os.getcwd", os.getcwd())


class dict2(object):
    def __init__(self, d):
        self.__dict__ = d



def ztest():
    import sklearn as sk
    print(sk)


####################################################################################################
def pd_col_to_onehot(dfref, colname=None, colonehot=None, return_val="dataframe,column"):
    """
    :param df:
    :param colname:
    :param colonehot: previous one hot columns
    :param returncol:
    :return:
    """
    df = copy.deepcopy(dfref)
    coladded = []
    colname = list(df.columns) if colname is None else colname

    # Encode each column into OneHot
    for x in colname:
        try:
            nunique = len(df[x].unique())
            print(x, nunique, df.shape, flush=True)

            if nunique > 2:
                df = pd.concat([df, pd.get_dummies(df[x], prefix=x)], axis=1).drop([x], axis=1)
            else:
                df[x] = df[x].factorize()[0]  # put into 0,1 format
            coladded.append(x)
        except Exception as e:
            print(x, e)

    # Add missing category columns
    if colonehot is not None:
        for x in colonehot:
            if not x in df.columns:
                df[x] = 0
                print(x, "added")
                coladded.append(x)

    colnew = colonehot if colonehot is not None else [c for c in df.columns if c not in colname]
    if return_val == "dataframe,param":
        return df[colnew], colnew

    else:
        return df[colnew]




def pd_colcat_mergecol(df, col_list, x0, colid="easy_id"):
    """
       Merge category onehot column
    :param df:
    :param l:
    :param x0:
    :return:
    """
    dfz = pd.DataFrame({colid: df[colid].values})
    for t in col_list:
        ix = t.rfind("_")
        val = int(t[ix + 1 :])
        print(ix, t[ix + 1 :])
        dfz[t] = df[t].apply(lambda x: val if x > 0 else 0)

    # print(dfz)
    dfz = dfz.set_index(colid)
    dfz[x0] = dfz.iloc[:, :].sum(1)
    for t in dfz.columns:
        if t != x0:
            del dfz[t]
    return dfz




def pd_colcat_tonum(df, colcat="all", drop_single_label=False, drop_fact_dict=True):
    """
    Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
    using the following logic:
    * categorical with only a single value will be marked as zero (or dropped, if requested)
    * categorical with two values will be replaced with the result of Pandas `factorize`
    * categorical with more than two values will be replaced with the result of Pandas `get_dummies`
    * numerical columns will not be modified
    **Returns:** DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
    else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
    value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
    present in the data-set
    Parameters
    ----------
    df : NumPy ndarray / Pandas DataFrame
        The data-set to encode
    colcat : sequence / string
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
        all columns are nominal. If None, nothing happens. Default: 'all'
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
    """
    df = convert(df, "dataframe")
    if colcat is None:
        return df
    elif colcat == "all":
        colcat = df.columns
    df_out = pd.DataFrame()
    binary_columns_dict = dict()

    for col in df.columns:
        if col not in colcat:
            df_out.loc[:, col] = df[col]

        else:
            unique_values = pd.unique(df[col])
            if len(unique_values) == 1 and not drop_single_label:
                df_out.loc[:, col] = 0
            elif len(unique_values) == 2:
                df_out.loc[:, col], binary_columns_dict[col] = pd.factorize(df[col])
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                df_out = pd.concat([df_out, dummies], axis=1)
    if drop_fact_dict:
        return df_out
    else:
        return df_out, binary_columns_dict




def pd_colcat_mapping(df, colname):
    """
     for col in colcat :
        df[col] = df[col].apply(lambda x : colcat_map["cat_map"][col].get(x)  )

    :param df:
    :param colname:
    :return:
    """
    mapping_rev = {
        col: {n: cat for n, cat in enumerate(df[col].astype("category").cat.categories)}
        for col in df[colname]
    }

    mapping = {
        col: {cat: n for n, cat in enumerate(df[col].astype("category").cat.categories)}
        for col in df[colname]
    }

    return {"cat_map": mapping, "cat_map_inverse": mapping_rev}



def pd_colcat_toint(dfref, colname, colcat_map=None, suffix=None):
    df = dfref[colname]
    suffix = "" if suffix is None else suffix
    colname_new = []
    
    if colcat_map is not None:
        for col in colname:
            ddict = colcat_map[col]["encode"]
            df[col + suffix], label = df[col].apply(lambda x: ddict.get(x))
            colname_new.append(col + suffix)
        
        return df[colname], colcat_map
    
    colcat_map = {}
    for col in colname:
        colcat_map[col] = {}
        df[col + suffix], label = df[col].factorize()
        colcat_map[col]["decode"] = {i: t for i, t in enumerate(list(label))}
        colcat_map[col]["encode"] = {t: i for i, t in enumerate(list(label))}
        colname_new.append(col + suffix)
    
    return df[colname_new], colcat_map


def pd_col_tocluster(
    df, colname=None, colexclude=None, colmodelmap=None, suffix="_bin",
    na_value=-1, return_val="dataframe,param",
    params = { "KMeans_n_clusters" : 8   , "KMeans_init": 'k-means++', "KMeans_n_init":10,
               "KMeans_max_iter" : 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances" : 'auto',
               "KMeans_verbose" : 0, "KMeans_random_state": None,
               "KMeans_copy_x": True, "KMeans_n_jobs" : None, "KMeans_algorithm" : 'auto'}
 ):

    """
    colbinmap = for each column, definition of bins
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
       :param df:
       :param method:
       :return:
    """

    colexclude = [] if colexclude is None else colexclude
    colname = colname if colname is not None else list(df.columns)
    colnew = []
    col_stat = OrderedDict()
    colmap = OrderedDict()

    #Bin Algo
    p = dict2(params)  # Bin  model params

    def bin_create_cluster(dfc):
        kmeans_model = KMeans(n_clusters= p.KMeans_n_clusters, init=p.KMeans_init, n_init=p.KMeans_n_init,
            max_iter=p.KMeans_max_iter, tol=p.KMeans_tol, precompute_distances=p.KMeans_precompute_distances,
            verbose=p.KMeans_verbose, random_state=p.KMeans_random_state,
            copy_x=p.KMeans_copy_x, n_jobs=p.KMeans_n_jobs, algorithm=p.KMeans_algorithm)

        kmeans_model.fit(dfc.values)
        return kmeans_model


    # Loop  on all columns
    for c in colname:
        if c in colexclude:
            continue
        print(c)
        df[c] = df[c].astype(np.float32)
        non_nan_index = np.where(~np.isnan(df[c]))[0]

        if colmodelmap is not None:
            model = colmodelmap.get(c)
        else:   
            model = bin_create_cluster(df.loc[non_nan_index][c].values.reshape((-1, 1)))

        cbin = c + suffix
        df.loc[non_nan_index][cbin] = model.predict(df.loc[non_nan_index][c].values.reshape((-1, 1))).reshape((-1,))


        # NA processing
        df[cbin] = df[cbin].astype("float")
        df[cbin] = df[cbin].apply(lambda x: x if x >= 0.0 else na_value)  # 3 NA Values
        df[cbin] = df[cbin].astype("int")
        col_stat = df.groupby(cbin).agg({c: {"size", "min", "mean", "max"}})
        colmap[c] = model
        colnew.append(cbin)

        print(col_stat)

    if return_val == "dataframe":
        return df[colnew]

    elif return_val == "param":
        return colmap
    else:
        return df[colnew], colmap


def pd_colnum_tocat(
    df, colname=None, colexclude=None, colbinmap=None, bins=5, suffix="_bin",
    method="uniform", na_value=-1, return_val="dataframe,param",
    params = { "KMeans_n_clusters" : 8   , "KMeans_init": 'k-means++', "KMeans_n_init":10,
               "KMeans_max_iter" : 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances" : 'auto',
               "KMeans_verbose" : 0, "KMeans_random_state": None,
               "KMeans_copy_x": True, "KMeans_n_jobs" : None, "KMeans_algorithm" : 'auto'}
 ):
    """
    colbinmap = for each column, definition of bins
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
       :param df:
       :param method:
       :return:
    """

    colexclude = [] if colexclude is None else colexclude
    colname = colname if colname is not None else list(df.columns)
    colnew = []
    col_stat = OrderedDict()
    colmap = OrderedDict()

    #Bin Algo
    # p = dict2(params)  # Bin  model params
    def bin_create(dfc, bins):
        mi, ma = dfc.min(), dfc.max()
        space = (ma - mi) / bins
        lbins = [mi + i * space for i in range(bins + 1)]
        lbins[0] -= 0.0001
        return lbins

    def bin_create_quantile(dfc, bins):
        qt_list_ref = np.arange(0, 1.00001, 1.0 / bins)
        # print(qt_list_ref )
        qt_list = dfc.quantile(qt_list_ref)
        # print(qt_list )
        lbins = list(qt_list.values)
        lbins[0] -= 0.01
        return lbins

    # def bin_create_cluster(dfc):
    #     kmeans = KMeans(n_clusters= p.KMeans_n_clusters, init=p.KMeans_init, n_init=p.KMeans_n_init,
    #         max_iter=p.KMeans_max_iter, tol=p.KMeans_tol, precompute_distances=p.KMeans_precompute_distances,
    #         verbose=p.KMeans_verbose, random_state=p.KMeans_random_state,
    #         copy_x=p.KMeans_copy_x, n_jobs=p.KMeans_n_jobs, algorithm=p.KMeans_algorithm).fit(dfc)
    #     return kmeans.predict(dfc)

    # Loop  on all columns
    for c in colname:
        if c in colexclude:
            continue
        print(c)
        df[c] = df[c].astype(np.float32)

        # Using Prebin Map data
        if colbinmap is not None:
            lbins = colbinmap.get(c)
        else:
            if method == "quantile":
                lbins = bin_create_quantile(df[c], bins)
            # elif method == "cluster":
            #     non_nan_index = np.where(~np.isnan(df[c]))[0]
            #     lbins = bin_create_cluster(df.loc[non_nan_index][c].values.reshape((-1, 1))).reshape((-1,))
            else:
                lbins = bin_create(df[c], bins)

        cbin = c + suffix
        # if method == 'cluster':
        #     df.loc[non_nan_index][cbin] = lbins
        # else:
        labels = np.arange(0, len(lbins) - 1)
        df[cbin] = pd.cut(df[c], bins=lbins, labels=labels)


        # NA processing
        df[cbin] = df[cbin].astype("float")
        df[cbin] = df[cbin].apply(lambda x: x if x >= 0.0 else na_value)  # 3 NA Values
        df[cbin] = df[cbin].astype("int")
        col_stat = df.groupby(cbin).agg({c: {"size", "min", "mean", "max"}})
        colmap[c] = lbins
        colnew.append(cbin)

        print(col_stat)

    if return_val == "dataframe":
        return df[colnew]

    elif return_val == "param":
        return colmap
    else:
        return df[colnew], colmap





def pd_colnum_normalize(df, colnum_log, colproba):
    """

    :param df:
    :param colnum_log:
    :param colproba:
    :return:
    """

    for x in colnum_log:
        try:
            df[x] = np.log(df[x].values.astype(np.float64) + 1.1)
            df[x] = df[x].replace(-np.inf, 0)
            df[x] = df[x].fillna(0)
            print(x, df[x].min(), df[x].max())
            df[x] = df[x] / df[x].max()
        except BaseException:
            pass

    for x in colproba:
        print(x)
        df[x] = df[x].replace(-1, 0.5)
        df[x] = df[x].fillna(0.5)

    return df




def pd_col_remove(df, cols):
    for x in cols:
        try:
            del df[x]
        except BaseException:
            pass
    return df


def pd_col_intersection(df1, df2, colid):
    """
    :param df1:
    :param df2:
    :param colid:
    :return :
    """
    n2 = list(set(df1[colid].values).intersection(df2[colid]))
    print("total matchin", len(n2), len(df1), len(df2))
    return n2


def pd_col_merge_onehot(df, colname):
    """
      Merge columns into single (hotn
    :param df:
    :param colname:
    :return :
    """
    dd = {}
    for x in colname:
        merge_array = []
        for t in df.columns:
            if x in t and t[len(x) : len(x) + 1] == "_":
                merge_array.append(t)
        dd[x] = merge_array
    return dd



def pd_col_to_num(df, colname=None, default=np.nan):
    def to_float(x):
        try:
            return float(x)
        except BaseException:
            return default

    colname = list(df.columns) if colname is None else colname
    for c in colname:
        df[c] = df[c].apply(lambda x: to_float(x))
    return df





def pd_col_filter(df, filter_val=None, iscol=1):
    """
   # Remove Columns where Index Value is not in the filter_value
   # filter1= X_client['client_id'].values
   :param df:
   :param filter_val:
   :param iscol:
   :return:
   """
    axis = 1 if iscol == 1 else 0
    col_delete = []
    for colname in df.index.values:  # !!!! row Delete
        if colname in filter_val:
            col_delete.append(colname)

    df2 = df.drop(col_delete, axis=axis, inplace=False)
    return df2





def pd_col_fillna(
    dfref,
    colname=None,
    method="frequent",
    value=None,
    colgroupby=None,
    return_val="dataframe,param",
):
    """
    Function to fill NaNs with a specific value in certain columns
    Arguments:
        df:            dataframe
        colname:      list of columns to remove text
        value:         value to replace NaNs with

    Returns:
        df:            new dataframe with filled values
    """
    colname = list(dfref.columns) if colname is None else colname
    df = dfref[colname]
    params = {"method": method, "na_value": {}}
    for col in colname:
        nb_nans = df[col].isna().sum()

        if method == "frequent":
            x = df[col].value_counts().idxmax()

        if method == "mode":
            x = df[col].mode()

        if method == "median":
            x = df[col].median()

        if method == "median_conditional":
            x = df.groupby(colgroupby)[col].transform("median")  # Conditional median

        value = x if value is None else value
        print(col, nb_nans, "replaceBY", value)
        params["na_value"][col] = value
        df[col] = df[col].fillna(value)

    if return_val == "dataframe,param":
        return df, params
    else:
        return df


def pd_col_fillna_advanced(
    dfref, colname=None, method="median", colname_na=None, return_val="dataframe,param"
):
    """
    Function to fill NaNs with a specific value in certain columns
    https://impyute.readthedocs.io/en/master/

    Arguments:
        df:            dataframe
        colname:      list of columns to remove text
        colname_na : target na coluns
        value:         value to replace NaNs with
    Returns:
        df:            new dataframe with filled values
     https://impyute.readthedocs.io/en/master/user_guide/overview.html

    """
    import impyute as impy

    colname = list(dfref.columns) if colname is None else colname
    df = dfref[colname]
    params = {"method": method, "na_value": {}}
    for col in colname:
        nb_nans = df[col].isna().sum()
        print(nb_nans)

    if method == "mice":
        from impyute.imputation.cs import mice

        imputed_df = mice(df.values)
        dfout = pd.DataFrame(data=imputed_df, columns=colname)

    elif method == "knn":
        from impyute.imputation.cs import fast_knn

        imputed_df = fast_knn(df.values, k=5)
        dfout = pd.DataFrame(data=imputed_df, columns=colname)

    if return_val == "dataframe,param":
        return dfout, params
    else:
        return dfout


def pd_col_fillna_datawig(
    dfref, colname=None, method="median", colname_na=None, return_val="dataframe,param"
):
    """
    Function to fill NaNs with a specific value in certain columns
    https://impyute.readthedocs.io/en/master/

    Arguments:
        df:            dataframe
        colname:      list of columns to remove text
        colname_na : target na coluns
        value:         value to replace NaNs with
    Returns:
        df:            new dataframe with filled values
     https://impyute.readthedocs.io/en/master/user_guide/overview.html

    """
    import impyute as impy

    colname = list(dfref.columns) if colname is None else colname
    df = dfref[colname]
    params = {"method": method, "na_value": {}}
    for colna in colname_na:
        nb_nans = df[colna].isna().sum()
        print(nb_nans)

    if method == "datawig":
        import datawig

        for colna in colname_na:
            imputer = datawig.SimpleImputer(
                input_columns=colname,
                output_column=colna,  # the column we'd like to impute values for
                output_path="preprocess_fillna/",  # stores model data and metrics
            )

            # Fit an imputer model on the train data
            imputer.fit(train_df=df)

            # Impute missing values and return original dataframe with predictions
            dfout = imputer.predict(df)

    if return_val == "dataframe,param":
        return dfout, params
    else:
        return dfout


def pd_row_drop_above_thresh(df, colnumlist, thresh):
    """
    Function to remove outliers above a certain threshold
    Arguments:
        df:     dataframe
        col:    col from which to remove outliers
        thresh: value above which to remove row
        colnumlist:list
    Returns:
        df:     dataframe with outliers removed
    """
    for col in colnumlist:
        df = df.drop(df[(df[col] > thresh)], axis=0)
    return df






def pd_pipeline_apply(df, pipeline):
    """
      pipe_preprocess_colnum = [
      (pd_col_to_num, {"val": "?", })
    , (pd_colnum_tocat, {"colname": None, "colbinmap": colnum_binmap, 'bins': 5,
                         "method": "uniform", "suffix": "_bin",
                         "return_val": "dataframe"})

    , (pd_col_to_onehot, {"colname": None, "colonehot": colnum_onehot,
                          "return_val": "dataframe"})
      ]
    :param df:
    :param pipeline:
    :return:
    """
    dfi = copy.deepcopy(df)
    for i, function in enumerate(pipeline):
        print(
            "############## Pipeline ", i, "Start", dfi.shape, str(function[0].__name__), flush=True
        )
        dfi = function[0](dfi, **function[1])
        print("############## Pipeline  ", i, "Finished", dfi.shape, flush=True)
    return dfi


def pd_df_sampling(df, coltarget="y", n1max=10000, n2max=-1, isconcat=1):
    """
        DownSampler
    :param df:
    :param coltarget: binary class
    :param n1max:
    :param n2max:
    :param isconcat:
    :return:
    """
    df1 = df[df[coltarget] == 0].sample(n=n1max)

    n2max = len(df[df[coltarget] == 1]) if n2max == -1 else n2max
    df0 = df[df[coltarget] == 1].sample(n=n2max)

    if isconcat:
        df2 = pd.concat((df1, df0))
        df2 = df2.sample(frac=1.0, replace=True)
        return df2

    else:
        print("y=1", n2max, "y=0", len(df1))
        return df0, df1


def pd_df_stack(df_list, ignore_index=True):
    """
    Concat vertically dataframe
    :param df_list:
    :return:
    """
    df0 = None
    for i, dfi in enumerate(df_list):
        if df0 is None:
            df0 = dfi
        else:
            try:
                df0 = df0.append(dfi, ignore_index=ignore_index)
            except Exception as e:
                print("Error appending: ", i, e)
    return df0



def pd_stat_correl_pair(df, coltarget=None, colname=None):
    """
      Genearte correletion between the column and target column
      df represents the dataframe comprising the column and colname comprising the target column
    :param df:
    :param colname: list of columns
    :param coltarget : target column

    :return:
    """
    from scipy.stats import pearsonr

    colname = colname if colname is not None else list(df.columns)
    target_corr = []
    for col in colname:
        target_corr.append(pearsonr(df[col].values, df[coltarget].values)[0])

    df_correl = pd.DataFrame({"colx": [""] * len(colname), "coly": colname, "correl": target_corr})
    df_correl[coltarget] = colname
    return df_correl





def pd_stat_colcheck(df):
    """
    :param df:
    :return :
    """
    for x in df.columns:
        if len(df[x].unique()) > 2 and df[x].dtype != np.dtype("O"):
            print(x, len(df[x].unique()), df[x].min(), df[x].max())



def pd_stat_jupyter_profile(df, savefile="report.html", title="Pandas Profile"):
    """ Describe the tables
        #Pandas-Profiling 2.0.0
        df.profile_report()
    """
    import pandas_profiling as pp

    print("start profiling")
    profile = df.profile_report(title=title)
    profile.to_file(output_file=savefile)
    colexclude = profile.get_rejected_variables(threshold=0.98)
    return colexclude


def pd_stat_distribution_colnum(df):
    """ Describe the tables


   """
    coldes = [
        "col",
        "coltype",
        "dtype",
        "count",
        "min",
        "max",
        "nb_na",
        "pct_na",
        "median",
        "mean",
        "std",
        "25%",
        "75%",
        "outlier",
    ]

    def getstat(col):
        """
         max, min, nb, nb_na, pct_na, median, qt_25, qt_75,
         nb, nb_unique, nb_na, freq_1st, freq_2th, freq_3th
         s.describe()
         count    3.0  mean     2.0 std      1.0
         min      1.0   25%      1.5  50%      2.0
         75%      2.5  max      3.0
      """
        ss = list(df[col].describe().values)
        ss = [str(df[col].dtype)] + ss
        nb_na = df[col].isnull().sum()
        ntot = len(df)
        ss = ss + [nb_na, nb_na / (ntot + 0.0)]

        return pd.Series(
            ss,
            ["dtype", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "nb_na", "pct_na"],
        )

    dfdes = pd.DataFrame([], columns=coldes)
    cols = df.columns
    for col in cols:
        dtype1 = str(df[col].dtype)
        if dtype1[0:3] in ["int", "flo"]:
            row1 = getstat(col)
            dfdes = pd.concat((dfdes, row1))

        if dtype1 == "object":
            pass



def pd_stat_histogram(df, bins=50, coltarget="diff"):
    """
    :param df:
    :param bins:
    :param coltarget:
    :return:
    """
    hh = np.histogram(
        df[coltarget].values, bins=bins, range=None, normed=None, weights=None, density=None
    )
    hh2 = pd.DataFrame({"bins": hh[1][:-1], "freq": hh[0]})
    hh2["density"] = hh2["freqall"] / hh2["freqall"].sum()
    return hh2


def pd_stat_histogram_groupby(df, bins=50, coltarget="diff", colgroupby="y"):
    """
    :param df:
    :param bins:
    :param coltarget:
    :param colgroupby:
    :return:
    """
    dfhisto = pd_stat_histogram_groupby(df, bins, coltarget)
    xunique = list(df[colgroupby].unique())

    # todo : issues with naming
    for x in xunique:
        dfhisto1 = pd_stat_histogram_groupby(df[df[colgroupby] == x], bins, coltarget)
        dfhisto = pd.concat((dfhisto, dfhisto1))

    return dfhisto


def pd_stat_na_perow(df, n=10 ** 6):
    """
    :param df:
    :param n:
    :return:
    """
    ll = []
    n = 10 ** 6
    for ii, x in df.iloc[:n, :].iterrows():
        ii = 0
        for t in x:
            if pd.isna(t) or t == -1:
                ii = ii + 1
        ll.append(ii)
    dfna_user = pd.DataFrame(
        {"": df.index.values[:n], "n_na": ll, "n_ok": len(df.columns) - np.array(ll)}
    )
    return dfna_user


def pd_stat_distribution(df, subsample_ratio=1.0):
    """
    :param df:
    :return:
    """
    print("Univariate distribution")
    ll = {
        x: []
        for x in [
            "col",
            "n",
            "n_na",
            "n_notna",
            "n_na_pct",
            "nunique",
            "nunique_pct",
            "xmin",
            "xmin_freq",
            "xmin_pct",
            "xmax",
            "xmax_freq",
            "xmax_pct",
            "xmed",
            "xmed_freq",
            "xmed_pct",
        ]
    }

    if subsample_ratio < 1.0:
        df = df.sample(frac=subsample_ratio)

    nn = len(df) + 0.0
    for x in df.columns:
        try:
            xmin = df[x].min()
            nx = len(df[df[x] < xmin + 0.01])  # Can failed if string
            ll["xmin_freq"].append(nx)
            ll["xmin"].append(xmin)
            ll["xmin_pct"].append(nx / nn)

            xmed = df[x].median()
            nx = len(df[(df[x] > xmed - 0.1) & (df[x] < xmed + 0.1)])
            ll["xmed_freq"].append(nx)
            ll["xmed"].append(xmed)
            ll["xmed_pct"].append(nx / nn)

            xmax = df[x].max()
            nx = len(df[df[x] > xmax - 0.01])
            ll["xmax_freq"].append(nx)
            ll["xmax"].append(xmax)
            ll["xmax_pct"].append(nx / nn)

            n_notna = df[x].count()
            ll["n_notna"].append(n_notna)
            ll["n_na"].append(nn - n_notna)
            ll["n"].append(nn)
            ll["n_na_pct"].append((nn - n_notna) / nn * 1.0)

            nx = df[x].nunique()
            ll["nunique"].append(nx)  # Should be in last
            ll["nunique_pct"].append(nx / nn)  # Should be in last
            ll["col"].append(x)  # Should be in last
        except Exception as e:
            print(x, e)

    # for k, x in ll.items():
    #    print(k, len(x))

    ll = pd.DataFrame(ll)
    return ll




def convert(data, to):
    """
    :param data:
    :param to:
    :return :
    """
    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == "list":
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == "dataframe":
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError("cannot handle data conversion of type: {} to {}".format(type(data), to))
    else:
        return converted





def col_stat_getcategorydict_freq(catedict):
    """ Generate Frequency of category : Id, Freq, Freqin%, CumSum%, ZScore
      given a dictionnary of category parsed previously
  """
    catlist = []
    for key, v in list(catedict.items()):
        df = pd.DataFrame(v)  # , ["category", "freq"])
        df["freq_pct"] = 100.0 * df["freq"] / df["freq"].sum()
        df["freq_zscore"] = df["freq"] / df["freq"].std()
        df = df.sort_values(by=["freq"], ascending=0)
        df["freq_cumpct"] = 100.0 * df["freq_pct"].cumsum() / df["freq_pct"].sum()
        df["rank"] = np.arange(0, len(df.index.values))
        catlist.append((key, df))
    return catlist



def col_extractname_colbin(cols2):
    """
    1hot column name to generic column names
    :param cols2:
    :return:
    """
    coln = []
    for ss in cols2:
        xr = ss[ss.rfind("_") + 1 :]
        xl = ss[: ss.rfind("_")]
        if len(xr) < 3:  # -1 or 1
            coln.append(xl)
        else:
            coln.append(ss)

    coln = np_drop_duplicates(coln)
    return coln


def col_getnumpy_indice(colall, colcat):
    def np_find_indice(v, x):
        for i, j in enumerate(v):
            if j == x:
                return i
        return -1

    return [np_find_indice(colall, x) for x in colcat]



def col_extractname(col_onehot):
    """
    Column extraction
    :param col_onehot
    :return:
    """
    colnew = []
    for x in col_onehot:
        if len(x) > 2:
            if x[-2] == "_":
                if x[:-2] not in colnew:
                    colnew.append(x[:-2])

            elif x[-2] == "-":
                if x[:-3] not in colnew:
                    colnew.append(x[:-3])

            else:
                if x not in colnew:
                    colnew.append(x)
    return colnew


def col_remove(cols, colsremove, mode="exact"):
    """
    Parameters
    ----------
    cols : TYPE
        DESCRIPTION.
    colsremove : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is "exact", "fuzzy"

    Returns
    -------
    cols : TYPE
        DESCRIPTION.  remove column name from list

    """
    if mode == "exact" :
      for x in colsremove:
         try:
            cols.remove(x)
         except BaseException:
            pass
      return cols

    if mode == "fuzzy" :
     cols3 = []
     for t in cols:
        flag = 0
        for x in colsremove:
            if x in t:
                flag = 1
                break
        if flag == 0:
            cols3.append(t)
     return cols3






####################################################################################################
"""
https://github.com/abhayspawar/featexp

from featexp import get_univariate_plots
get_univariate_plots(data=data_train, target_col='target', data_test=data_test, features_list=['DAYS_EMPLOYED'])

# data_test and features_list are optional. 
# Draws plots for all columns if features_list not passed
# Draws only train data plots if no test_data passed


from featexp import get_trend_stats_feature
stats = get_trend_stats(data=data_train, target_col='target', data_test=data_test)

# data_test is optional. If not passed, trend correlations aren't calculated.


"""
def pd_colnum_tocat_stat(df, feature, target_col, bins, cuts=0):
    """
    Bins continuous features into equal sample size buckets and returns the target mean in each bucket. Separates out
    nulls into another bucket.
    :param df: dataframe containg features and target column
    :param feature: feature column name
    :param target_col: target column
    :param bins: Number bins required
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
    :return: If cuts are passed only df_grouped data is returned, else cuts and df_grouped data is returned
    """
    has_null = pd.isnull(df[feature]).sum() > 0
    if has_null == 1:
        data_null = df[pd.isnull(df[feature])]
        df = df[~pd.isnull(df[feature])]
        df.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts == 0:
        is_train = 1
        prev_cut = min(df[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(df[feature], i * 100 / bins)
            if next_cut > prev_cut + .000001:  # float numbers shold be compared with some threshold!
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        # if reduced_cuts>0:
        #     print('Reduced the number of bins due to less variation in feature')
        cut_series = pd.cut(df[feature], cuts)
    else:
        cut_series = pd.cut(df[feature], cuts)

    df_grouped = df.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]})
    df_grouped.columns = ['_'.join(cols).strip() for cols in df_grouped.columns.values]
    df_grouped[df_grouped.index.name] = df_grouped.index
    df_grouped.reset_index(inplace=True, drop=True)
    df_grouped = df_grouped[[feature] + list(df_grouped.columns[0:3])]
    df_grouped = df_grouped.rename(index=str, columns={target_col + '_size': 'Samples_in_bin'})
    df_grouped = df_grouped.reset_index(drop=True)
    corrected_bin_name = '[' + str(min(df[feature])) + ', ' + str(df_grouped.loc[0, feature]).split(',')[1]
    df_grouped[feature] = df_grouped[feature].astype('category')
    df_grouped[feature] = df_grouped[feature].cat.add_categories(corrected_bin_name)
    df_grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = df_grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype('category')
        grouped_null[feature] = grouped_null[feature].cat.add_categories('Nulls')
        grouped_null.loc[0, feature] = 'Nulls'
        grouped_null.loc[0, 'Samples_in_bin'] = len(data_null)
        grouped_null.loc[0, target_col + '_mean'] = data_null[target_col].mean()
        grouped_null.loc[0, feature + '_mean'] = np.nan
        df_grouped[feature] = df_grouped[feature].astype('str')
        df_grouped = pd.concat([grouped_null, df_grouped], axis=0)
        df_grouped.reset_index(inplace=True, drop=True)

    df_grouped[feature] = df_grouped[feature].astype('str').astype('category')
    if is_train == 1:
        return (cuts, df_grouped)
    else:
        return (df_grouped)




def pd_stat_shift_trend_changes(df, feature, target_col, threshold=0.03):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    :param df: df_grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return: number of trend chagnes for the feature
    """
    df = df.loc[df[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs = df[target_col + '_mean'].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = df[target_col + '_mean'].max() - df[target_col + '_mean'].min()
    target_diffs_mod = target_diffs.fillna(0).abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return (tot_trend_changes)



def pd_stat_shift_trend_correlation(df, df_test, colname, target_col):
    """
    Calculates correlation between train and test trend of colname wrt target.
    :param df: train df data
    :param df_test: test df data
    :param colname: colname column name
    :param target_col: target column name
    :return: trend correlation between train and test
    """
    df = df[df[colname] != 'Nulls'].reset_index(drop=True)
    df_test = df_test[df_test[colname] != 'Nulls'].reset_index(drop=True)

    if df_test.loc[0, colname] != df.loc[0, colname]:
        df_test[colname] = df_test[colname].cat.add_categories(df.loc[0, colname])
        df_test.loc[0, colname] = df.loc[0, colname]
    df_test_train = df.merge(df_test[[colname, target_col + '_mean']], on=colname,
                                       how='left',
                                       suffixes=('', '_test'))
    nan_rows = pd.isnull(df_test_train[target_col + '_mean']) | pd.isnull(
        df_test_train[target_col + '_mean_test'])
    df_test_train = df_test_train.loc[~nan_rows, :]
    if len(df_test_train) > 1:
        trend_correlation = np.corrcoef(df_test_train[target_col + '_mean'],
                                        df_test_train[target_col + '_mean_test'])[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + colname + ". Correlation can't be calculated")

    return (trend_correlation)



def pd_stat_shift_changes(df, target_col, features_list=0, bins=10, df_test=0):
    """
    Calculates trend changes and correlation between train/test for list of features
    :param df: dfframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous colname
    :param df_test: test df which has to be compared with input df for correlation
    :return: dfframe with trend changes and trend correlation (if test df passed)
    """

    if type(features_list) == int:
        features_list = list(df.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(df_test) == pd.core.frame.DataFrame
    ignored = []
    for colname in features_list:
        if df[colname].dtype == 'O' or colname == target_col:
            ignored.append(colname)
        else:
            cuts, df_grouped = pd_colnum_tocat_stat(df=df, colname=colname, target_col=target_col, bins=bins)
            trend_changes = pd_stat_shift_trend_correlation(df=df_grouped, colname=colname, target_col=target_col)
            if has_test:
                df_test = pd_colnum_tocat_stat(df=df_test.reset_index(drop=True), colname=colname,
                                                target_col=target_col, bins=bins, cuts=cuts)
                trend_corr = pd_stat_shift_trend_correlation(df_grouped, df_test, colname, target_col)
                trend_changes_test = pd_stat_shift_changes(df=df_test, colname=colname,
                                                       target_col=target_col)
                stats = [colname, trend_changes, trend_changes_test, trend_corr]
            else:
                stats = [colname, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = ['colname', 'Trend_changes'] if has_test == False else ['colname', 'Trend_changes',
                                                                                   'Trend_changes_test',
                                                                                   'Trend_correlation']
    if len(ignored) > 0:
        print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')

    print('Returning stats for all numeric features')
    return (stats_all_df)



