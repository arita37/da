# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas



"""
import copy
import os
from collections import Counter
from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
import scipy as sci


try :
    import pandas_profiling

except Exception as e :
    print(e)



print("os.getcwd", os.getcwd())



####################################################################################################
####################################################################################################
def pd_col_to_onehot(df, colname):
    """
    :param df:
    :param colname:
    :return:
    """
    for x in colname:
        try:
            nunique = len(df[x].unique())
            print(x, nunique, df.shape, flush=True)

            if nunique > 2:
                df = pd.concat([df, pd.get_dummies(df[x], prefix=x)], axis=1).drop([x], axis=1)
                # df = pd.get_dummies(df, columns=cat_col, dummy_na=nan_as_category)
                # coli =   [ x +'_' + str(t) for t in  lb.classes_ ]
                # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
                # del df[x]
            else:
                lb = preprocessing.LabelBinarizer()
                vv = lb.fit_transform(df[x])
                df[x] = vv
        except Exception as e:
            print(x, e)
    # new_columns = [c for c in df.columns if c not in original_columns]
    return df


def pd_colnum_tocat(df, colname=None, colexclude=None,  suffix="_bin", method=""):
    """
    Preprocessing.KBinsDiscretizer([n_bins, …])	Bin continuous data into intervals.

    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    :param df:
    :param method:
    :return:
    """
    colname = colname if colname is not None else list(df.columns)
    colnew  = []
    for c in df.columns:
        if c in colexclude:
            continue

        df[c] = df[c].astype(np.float32)
        mi, ma = df[c].min(), df[c].max()
        space = (ma - mi) / 5
        bins = [mi + i * space for i in range(6)]
        bins[0] -= 0.0000001

        labels = np.arange(0, len(bins))
        colnew.append( c + suffix )
        df[c + suffix ] = pd.cut(df[c], bins=bins, labels=labels)
    return df


def pd_colnum_tocat2(df, colname=None, colexclude=None, suffix="_bin",
                     method="uniform", bins=None):
    """
    preprocessing.KBinsDiscretizer([n_bins, …])	Bin continuous data into intervals.

    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    :param df:
    :param method: uniform,
    :return:

    strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, (default=’quantile’)
    Strategy used to define the widths of the bins.

    uniform
    All bins in each feature have identical widths.

    quantile
    All bins in each feature have the same number of points.

    kmeans
    Values in each bin have the same nearest center of a 1D k-means cluster.

    https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html#sphx-glr-auto-examples-preprocessing-plot-discretization-classification-py

    """
    colname = colname if colname is not None else list(df.columns)

    # transform the dataset with KBinsDiscretizer
    from sklearn.preprocessing import KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=method)

    for c in df.columns:
        if c in colexclude:
            continue
        df[c] = df[c].astype(np.float32)

        X_binned = enc.fit_transform(df[c].values)
        df[c + suffix ] = X_binned

    return df






def pd_col_merge(df, ll0):
    """
    :param df:
    :param ll0:
    :return :
    """
    dd = {}
    for x in ll0:
        ll2 = []
        for t in df.columns:
            if x in t and t[len(x) : len(x) + 1] == "_":
                ll2.append(t)
        dd[x] = ll2
    return dd


def pd_col_merge2(df, l, x0):
    """
    :param df:
    :param l:
    :param x0:
    :return:
    """
    dfz = pd.DataFrame({"easy_id": df["easy_id"].values})
    for t in l:
        ix = t.rfind("_")
        val = int(t[ix + 1 :])
        print(ix, t[ix + 1 :])
        dfz[t] = df[t].apply(lambda x: val if x > 0 else 0)

    # print(dfz)
    dfz = dfz.set_index("easy_id")
    dfz[x0] = dfz.iloc[:, :].sum(1)
    for t in dfz.columns:
        if t != x0:
            del dfz[t]
    return dfz


def pd_df_sampling(df, coltarget="y", n1max=10000, n2max=-1, isconcat=1):
    """
        DownSampler
    :param df:
    :param coltarget:
    :param n1max:
    :param n2max:
    :param isconcat:
    :return:
    """
    # n0  = len( df1 )
    # l1  = np.random.choice( len(df1) , n1max, replace=False)

    df1 = df[df[coltarget] == 0].sample(n=n1max)

    # df1   = df[ df[coltarget] == 1 ]
    # n1    = len(df1 )
    # print(n1)
    n2max = len(df[df[coltarget] == 1]) if n2max == -1 else n2max
    # l1    = np.random.choice(len(df1) , n2max, replace=False)
    df0 = df[df[coltarget] == 1].sample(n=n2max)
    # print(len(df0))

    if isconcat:
        df2 = pd.concat((df1, df0))
        df2 = df2.sample(frac=1.0, replace=True)
        return df2

    else:
        print("y=1", n2max, "y=0", len(df1))
        return df0, df1


def pd_stat_histogram(df, bins=50, coltarget="diff"):
    """
    :param df:
    :param bins:
    :param coltarget:
    :param colgroupby:
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


def pd_stat_na_percol(df):
    """
    :param df:
    :return:
    """
    ll = []
    for x in df.columns:
        nn = df[x].isnull().sum()
        nn = nn + len(df[df[x] == -1])

        ll.append(nn)
    dfna_col = pd.DataFrame({"col": list(df.columns), "n_na": ll})
    dfna_col["n_tot"] = len(df)
    dfna_col["pct_na"] = dfna_col["n_na"] / dfna_col["n_tot"]
    return dfna_col


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


def pd_stat_col_imbalance(df):
    """
    :param df:
    :return:
    """
    ll = {
        x: []
        for x in [
            "col", "xmin_freq", "nunique", "xmax_freq", "xmax", "xmin", "n", "n_na",
            "n_notna",]
    }

    nn = len(df)
    for x in df.columns:
        try:
            xmin = df[x].min()
            ll["xmin_freq"].append(len(df[df[x] < xmin + 0.01]))
            ll["xmin"].append(xmin)

            xmax = df[x].max()
            ll["xmax_freq"].append(len(df[df[x] > xmax - 0.01]))
            ll["xmax"].append(xmax)

            n_notna = df[x].count()
            ll["n_notna"].append(n_notna)
            ll["n_na"].append(nn - n_notna)
            ll["n"].append(nn)

            ll["nunique"].append(df[x].nunique())
            ll["col"].append(x)
        except:
            pass

    ll = pd.DataFrame(ll)
    ll["xmin_ratio"] = ll["xmin_freq"] / nn
    ll["xmax_ratio"] = ll["xmax_freq"] / nn
    return ll


import math


def np_conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    **Returns:** float
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def np_cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = sci.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def np_theils_u(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    s_xy = np_conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = sci.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def np_correlation_ratio(cat_array, num_array):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it possible to know which category is it
    associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
    a category can be determined with absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    cat_array : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    num_array : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    """
    cat_array = convert(cat_array, "array")
    num_array = convert(num_array, "array")
    fcat, _ = pd.factorize(cat_array)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = num_array[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(num_array, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def pd_num_correl_associations(
    df, colcat=None, mark_columns=False, theil_u=False, plot=True, return_results=False, **kwargs
):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    **Returns:** a DataFrame of the correlation/strength-of-association between all features
    **Example:** see `associations_example` under `dython.examples`
    Parameters
    ----------
    df : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    colcat : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by colcat
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    plot : Boolean, default = True
        If True, plot a heat-map of the correlation matrix
    return_results : Boolean, default = False
        If True, the function will return a Pandas DataFrame of the computed associations
    kwargs : any key-value pairs
        Arguments to be passed to used function and methods
    """
    df = convert(df, "dataframe")
    col = df.columns
    if colcat is None:
        colcat = list()
    elif colcat == "all":
        colcat = col
    corr = pd.DataFrame(index=col, columns=col)
    for i in range(0, len(col)):
        for j in range(i, len(col)):
            if i == j:
                corr[col[i]][col[j]] = 1.0
            else:
                if col[i] in colcat:
                    if col[j] in colcat:
                        if theil_u:
                            corr[col[j]][col[i]] = np_theils_u(df[col[i]], df[col[j]])
                            corr[col[i]][col[j]] = np_theils_u(df[col[j]], df[col[i]])
                        else:
                            cell = np_cramers_v(df[col[i]], df[col[j]])
                            corr[col[i]][col[j]] = cell
                            corr[col[j]][col[i]] = cell
                    else:
                        cell = np_correlation_ratio(df[col[i]], df[col[j]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                else:
                    if col[j] in colcat:
                        cell = np_correlation_ratio(df[col[j]], df[col[i]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                    else:
                        cell, _ = sci.pearsonr(df[col[i]], df[col[j]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = [
            "{} (nom)".format(col) if col in colcat else "{} (con)".format(col) for col in col
        ]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        pass
        """
        plt.figure(figsize=kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'))
        plt.show()
        """
    if return_results:
        return corr


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
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in df.columns:
        if col not in colcat:
            converted_dataset.loc[:, col] = df[col]
        else:
            unique_values = pd.unique(df[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:, col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:, col], binary_columns_dict[col] = pd.factorize(df[col])
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset, dummies], axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict


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


##### Get Kaiso limit ###############################################################
def fun_get_segmentlimit(x, l1):
    """
    :param x:
    :param l1:
    :return :
    """
    for i in range(0, len(l1)):
        if x >= l1[i]:
            return i + 1
    return i + 1 + 1


def np_drop_duplicates(l1):
    """
    :param l1:
    :return :
    """
    l0 = list(OrderedDict((x, True) for x in l1).keys())
    return l0


def col_extractname_colbin(cols2):
    """
    1hot column name to generic column names
    :param cols2:
    :return:
    """
    coln = []
    for ss in cols2:
        xr = ss[sci.rfind("_") + 1 :]
        xl = ss[: sci.rfind("_")]
        if len(xr) < 3:  # -1 or 1
            coln.append(xl)
        else:
            coln.append(ss)

    coln = np_drop_duplicates(coln)
    return coln


def pd_stat_col(df):
    """
    :param df:
    :return :
    """
    ll = {"col": [], "nunique": []}
    for x in df.columns:
        ll["col"].append(x)
        ll["nunique"].append(df[x].nunique())
    ll = pd.DataFrame(ll)
    n = len(df) + 0.0
    ll["ratio"] = ll["nunique"] / n
    ll["coltype"] = ll["nunique"].apply(lambda x: "cat" if x < 100 else "num")

    return ll


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


def pd_col_normalize(df, colnum_log, colproba):
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
        except:
            pass

    for x in colproba:
        print(x)
        df[x] = df[x].replace(-1, 0.5)
        df[x] = df[x].fillna(0.5)

    return df


def pd_col_check(df):
    """
    :param df:
    :return :
    """
    for x in df.columns:
        if len(df[x].unique()) > 2 and df[x].dtype != np.dtype("O"):
            print(x, len(df[x].unique()), df[x].min(), df[x].max())


def pd_col_remove(df, cols):
    for x in cols:
        try:
            del df[x]
        except:
            pass
    return df


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


def col_remove(cols, colsremove):
    # cols = list(df1.columns)
    """
    colsremove = [
    'y', 'def',
    'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
    'score' ,   'segment2' ,
    'scoreb', 'score_kaisob', 'segmentb', 'def_test'
    ]
    colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    """
    for x in colsremove:
        try:
            cols.remove(x)
        except:
            pass
    return cols


def col_remove_fuzzy(cols, colsremove):
    # cols = list(df1.columns)
    """
    :param cols:
    :param colsremove:
    :return:

      colsremove = [
         'y', 'def',
         'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
         'score' ,   'segment2' ,
         'scoreb', 'score_kaisob', 'segmentb', 'def_test'
      ]
      colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    """
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


def pd_col_filter(dfxx, cols):
    """
    :param dfxx:
    :param cols:
    :return:
    """
    df1 = copy.deepcopy(dfxx[cols + ["def", "y"]])
    df1 = df1[df1["def"] < 201905]
    df1 = df1[(df1["def"] > 201703) | (df1["def"] == -1)]
    return df1


def col_study_getcategorydict_freq(catedict):
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



def pd_num_correl_pair(df, coltarget=None, colname=None):
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
        target_corr.append( pearsonr(df[col].values, df[coltarget].values)[0])

    df_correl = pd.DataFrame({ "colx": [""]*len(colname),
                               "coly": colname,
                               "correl": target_corr  })
    df_correl[coltarget] = colname
    return df_correl






def pd_col_study_summary(df, colname, isprint=0):
    """

    :param df:
    :param colname:
    :param isprint:
    :return:
    """
    Xmat = df[colname].values
    n, m = np.shape(Xmat)

    colanalysis = []
    for icol, x in enumerate(colname):
        Xraw_1unique = np.unique(Xmat[:, icol])
        vv = [
            colname[icol],
            icol,
            len(Xraw_1unique),
            np.min(Xraw_1unique),
            np.max(Xraw_1unique),
            np.median(Xraw_1unique),
            np.mean(Xraw_1unique),
            np.std(Xraw_1unique),
        ]
        colanalysis.append(vv)

    colanalysis = pd.DataFrame(
        colanalysis,
        columns=[
            "Col_name",
            "Col_id",
            "Nb_Unique",
            "MinVal",
            "MaxVal",
            "MedianVal",
            "MeanVal",
            "StdDev",
        ],
    )
    if isprint:
        print(("Nb_Samples:", np.shape(Xmat)[0], "Nb Col:", len(colname)))
        print(colanalysis)
    return colanalysis


def pd_col_filter2(df_client_product, filter_val=[], iscol=1):
    """
   # Remove Columns where Index Value is not in the filter_value
   # filter1= X_client['client_id'].values
   :param df_client_product:
   :param filter_val:
   :param iscol:
   :return:
   """
    axis = 1 if iscol == 1 else 0
    col_delete1 = []
    for colname in df_client_product.index.values:  # !!!! row Delete
        if colname in filter_val:
            col_delete1.append(colname)

    df2 = df_client_product.drop(col_delete1, axis=axis, inplace=False)
    return df2



def pd_jupyter_profile(df):
    """ Describe the tables
        #Pandas-Profiling 2.0.0
        df.profile_report()
    """
    import pandas_profiling
    df.profile_report()




def pd_stat_describe(df):
    """ Describe the tables


   """
    coldes = [
        "col", "coltype", "dtype", "count", "min", "max", "nb_na", "pct_na", "median",
        "mean", "std", "25%", "75%", "outlier",
    ]

    def getstat(col, type1="num"):
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
            row1 = getstat(col, "num")
            dfdes = pd.concat((dfdes, row1))

        if dtype1 == "object":
            pass


def pd_df_stack(df_list):
    """
    :param df_list:
    :return:
    """
    df0 = None
    for i, dfi in enumerate(df_list):
        if df0 is None:
            df0 = dfi
        else:
            try:
                df0 = df0.append(dfi, ignore_index=True)
            except:
                print(("Error appending: " + str(i)))
    return df0


def pd_validation_struct():
    pass
    """
  https://github.com/jnmclarty/validada

  https://github.com/ResidentMario/checkpoints


  """


######################  Transformation   ###########################################################
def pd_colcat_label_toint(df):
    """
    # ["paris", "paris", "tokyo", "amsterdam"]  --> 2 ,5,6
    # np.array(le.inverse_transform([2, 2, 1]))
    le = preprocessing.LabelEncoder()
    le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    """
    Xmat = df.values
    le = sk.preprocessing.LabelEncoder()
    ncol = Xmat.shape[1]
    Xnew = np.zeros_like(Xmat)
    mapping_cat_int = {}

    for k in range(0, ncol):
        Xnew[:, k] = le.fit_transform(Xmat[:, k])
        mapping_cat_int[k] = le.get_params()

    return Xnew, mapping_cat_int







'''
def pd_stat_na_missing_show():
    #https://blog.modeanalytics.com/python-data-visualization-libraries/
    #Missing Data
    #missingno
   
    import missingno as msno
    %matplotlib inline
    msno.matrix(collisions.sample(250))
    
    null_pattern = (np.random.random(1000).reshape((50, 20)) > 0.5).astype(bool)
    null_pattern = pd.DataFrame(null_pattern).replace({False: None})
    msno.matrix(null_pattern.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ')
    #msno.bar is a simple visualization of nullity by column
    msno.bar(collisions.sample(1000))
    #The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another
    msno.heatmap(collisions)
    #At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be completely populated, while geographic information seems mostly complete, but spottier.
    #The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum rows.This visualization will comfortably accommodate up to 50 labelled variables. Past that range labels begin to overlap or become unreadable, and by default large displays omit them.
    #https://github.com/ResidentMario/missingno
    
    # dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap
    msno.dendrogram(collisions)
    #geographic distribution. 
    msno.geoplot(collisions, x='LONGITUDE', y='LATITUDE')
    msno.geoplot(collisions, x='LONGITUDE', y='LATITUDE', by='ZIP CODE')
'''


















