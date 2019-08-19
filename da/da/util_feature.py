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


def pd_colnum_tocat(
    df, colname=None, colexclude=None, colbinmap=None, bins=5, suffix="_bin",
    method="uniform", return_val="dataframe,param",
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
            else:
                lbins = bin_create(df[c], bins)

        cbin = c + suffix
        labels = np.arange(0, len(lbins) - 1)
        df[cbin] = pd.cut(df[c], bins=lbins, labels=labels)

        # NA processing
        df[cbin] = df[cbin].astype("int")
        df[cbin] = df[cbin].apply(lambda x: x if x >= 0.0 else -1)  # 3 NA Values
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


def np_correl_cat_cat_cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013):
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
    chi2 = sci.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def np_correl_cat_cat_theils_u(x, y):
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
    s_x = sci.stats.entropy(p_x)
    if s_x == 0:
        return 1
    return (s_x - s_xy) / s_x


def np_correl_cat_num_ratio(cat_array, num_array):
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
    cat_array : list / NumPy ndarray / Pandas Series   A sequence of categorical measurements
    num_array : list / NumPy ndarray / Pandas Series  A sequence of continuous measurements
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
    # df = convert(df, "dataframe")
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
                            corr[col[j]][col[i]] = np_correl_cat_cat_theils_u(
                                df[col[i]], df[col[j]]
                            )
                            corr[col[i]][col[j]] = np_correl_cat_cat_theils_u(
                                df[col[j]], df[col[i]]
                            )
                        else:
                            cell = np_correl_cat_cat_cramers_v(df[col[i]], df[col[j]])
                            corr[col[i]][col[j]] = cell
                            corr[col[j]][col[i]] = cell
                    else:
                        cell = np_correl_cat_num_ratio(df[col[i]], df[col[j]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                else:
                    if col[j] in colcat:
                        cell = np_correl_cat_num_ratio(df[col[j]], df[col[i]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                    else:
                        cell, _ = sci.stats.pearsonr(df[col[i]], df[col[j]])
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


def pd_stat_colcheck(df):
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
        except BaseException:
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
      remove column name from list
    """
    for x in colsremove:
        try:
            cols.remove(x)
        except BaseException:
            pass
    return cols


def col_remove_fuzzy(cols, colsremove):
    # cols = list(df1.columns)
    """
    :param cols:
    :param colsremove:
    :return:

      Remove column from Fuzzy matching.
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
        target_corr.append(pearsonr(df[col].values, df[coltarget].values)[0])

    df_correl = pd.DataFrame({"colx": [""] * len(colname), "coly": colname, "correl": target_corr})
    df_correl[coltarget] = colname
    return df_correl


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
