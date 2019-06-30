"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas

"""
# -*- coding: utf-8 -*-


import copy
import os
from collections import OrderedDict
from collections import Counter

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing

try:
    from catboost import CatBoostClassifier, Pool, cv
except Exception as e:
    print(e)


# import util


print('os.getcwd', os.getcwd())



########################################################################################################################
########################################################################################################################
def pd_col_to_onehot(df, colname):
    for x in colname:
        try:
            nunique = len(df[x].unique())
            print(x, nunique, df.shape, flush=True)

            if nunique > 2:
                df = pd.concat([df, pd.get_dummies(df[x], prefix=x)], axis=1).drop([x], axis=1)
                # coli =   [ x +'_' + str(t) for t in  lb.classes_ ]
                # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
                # del df[x]
            else:
                lb = preprocessing.LabelBinarizer()
                vv = lb.fit_transform(df[x])
                df[x] = vv
        except Exception as e:
            print(x, e)
    return df


"""
def pd_col_to_onehot(df, nan_as_category=True, categorical_columns=None):

    #     categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    cat_col = [col for col in df.columns]
    df = pd.get_dummies(df, columns=cat_col, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
"""



def pd_num_tocat(df, colname = None,  colexclude=None, method=""):
    """
    :param df:
    :param method:
    :return:
    """
    colname = colname if colname is not None else list( df.columns)
    for c in df.columns:
        if c in colexclude:
            continue
        df[c]  = df[c].astype(np.float32)
        mi, ma = df[c].min(), df[c].max()
        space  = (ma - mi) / 5
        bins   = [mi + i * space for i in range(6)]
        bins[0] -= 0.0000001

        labels = np.arange(0, len(bins))
        df[c] = pd.cut(df[c], bins=bins, labels=labels)
    return df




def pd_col_merge(dfm3, ll0):
    dd = {}
    for x in ll0:
        ll2 = []
        for t in dfm3.columns:
            if x in t and t[len(x):len(x) + 1] == "_":
                ll2.append(t)
        dd[x] = ll2
    return dd


def pd_col_merge2(dfm3, l, x0):
    dfz = pd.DataFrame({'easy_id': dfm3['easy_id'].values})
    for t in l:
        ix = t.rfind("_")
        val = int(t[ix + 1:])
        print(ix, t[ix + 1:])
        dfz[t] = dfm3[t].apply(lambda x: val if x > 0 else 0)

    # print(dfz)
    dfz = dfz.set_index('easy_id')
    dfz[x0] = dfz.iloc[:, :].sum(1)
    for t in dfz.columns:
        if t != x0:
            del dfz[t]
    return dfz


def pd_sampling(df, coltarget="y", n1max=10000, n2max=-1, isconcat=1):
    """
    DownSampler      
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


#### Histo
def pd_stat_histo(dfm2, bins=50, col0='diff', col1='y'):
    hh = np.histogram(dfm2[col0].values,
                      bins=bins, range=None, normed=None, weights=None, density=None)
    hh2 = pd.DataFrame({'xall': hh[1][:-1],
                        'freqall': hh[0]})[['xall', 'freqall']]
    hh2['densityall'] = hh2['freqall'] / hh2['freqall'].sum()

    hh = np.histogram(dfm2[dfm2[col1] == 0][col0].values,
                      bins=bins, range=None, normed=None, weights=None, density=None)
    hh2['x0'] = hh[1][:-1]
    hh2['freq0'] = hh[0]
    hh2['density0'] = hh2['freq0'] / hh2['freq0'].sum()

    hh = np.histogram(dfm2[dfm2[col1] == 1][col0].values,
                      bins=bins, range=None, normed=None, weights=None, density=None)
    hh2['x1'] = hh[1][:-1]
    hh2['freq1'] = hh[0]
    hh2['density1'] = hh2['freq1'] / hh2['freq1'].sum()

    return hh2


def pd_stat_na_percol(dfm2):
    ll = []
    for x in dfm2.columns:
        nn = dfm2[x].isnull().sum()
        nn = nn + len(dfm2[dfm2[x] == -1])

        ll.append(nn)
    dfna_col = pd.DataFrame({'col': list(dfm2.columns), 'n_na': ll})
    dfna_col['n_tot'] = len(dfm2)
    dfna_col['pct_na'] = dfna_col['n_na'] / dfna_col['n_tot']
    return dfna_col


def pd_stat_na_perow(dfm2, n=10 ** 6):
    ll = [];
    n = 10 ** 6
    for ii, x in dfm2.iloc[:n, :].iterrows():
        ii = 0
        for t in x:
            if pd.isna(t) or t == -1:
                ii = ii + 1
        ll.append(ii)
    dfna_user = pd.DataFrame({'': dfm2.index.values[:n], 'n_na': ll,
                              'n_ok': len(dfm2.columns) - np.array(ll)})
    return dfna_user


def pd_stat_col_imbalance(df):
    ll = {x: [] for x in ['col', 'xmin_freq', 'nunique', 'xmax_freq', 'xmax',
                          'xmin', 'n', 'n_na', 'n_notna']}

    nn = len(df)
    for x in df.columns:
        try:
            xmin = df[x].min()
            ll['xmin_freq'].append(len(df[df[x] < xmin + 0.01]))
            ll['xmin'].append(xmin)

            xmax = df[x].max()
            ll['xmax_freq'].append(len(df[df[x] > xmax - 0.01]))
            ll['xmax'].append(xmax)

            n_notna = df[x].count()
            ll['n_notna'].append(n_notna)
            ll['n_na'].append(nn - n_notna)
            ll['n'].append(nn)

            ll['nunique'].append(df[x].nunique())
            ll['col'].append(x)
        except:
            pass

    ll = pd.DataFrame(ll)
    ll['xmin_ratio'] = ll['xmin_freq'] / nn
    ll['xmax_ratio'] = ll['xmax_freq'] / nn
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
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
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
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


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
    s_xy = np_conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def np_correlation_ratio(categories, measurements):
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
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    """
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def pd_correl_associations(df, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                 return_results = False, **kwargs):
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
    nominal_columns : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    plot : Boolean, default = True
        If True, plot a heat-map of the correlation matrix
    return_results : Boolean, default = False
        If True, the function will return a Pandas DataFrame of the computed associations
    kwargs : any key-value pairs
        Arguments to be passed to used function and methods
    """
    df = convert(df, 'dataframe')
    columns = df.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = np_theils_u(df[columns[i]], df[columns[j]])
                            corr[columns[i]][columns[j]] = np_theils_u(df[columns[j]], df[columns[i]])
                        else:
                            cell = np_cramers_v(df[columns[i]], df[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = np_correlation_ratio(df[columns[i]], df[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = np_correlation_ratio(df[columns[j]], df[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(df[columns[i]], df[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
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


def pd_cat_tonum(df, nominal_columns='all', drop_single_label=False, drop_fact_dict=True):
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
    nominal_columns : sequence / string
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
        all columns are nominal. If None, nothing happens. Default: 'all'
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
    """
    df = convert(df, 'dataframe')
    if nominal_columns is None:
        return df
    elif nominal_columns == 'all':
        nominal_columns = df.columns
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in df.columns:
        if col not in nominal_columns:
            converted_dataset.loc[:,col] = df[col]
        else:
            unique_values = pd.unique(df[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:,col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:,col], binary_columns_dict[col] = pd.factorize(df[col])
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset,dummies],axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict



def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted






#### Calculate KAISO Limit  #########################################################
def pd_segment_limit(dfm2, col_score='scoress', coldefault="y", ntotal_default=491, def_list=None, nblock=20.0):
    if def_list is None:
        def_list = np.ones(21) * ntotal_default / nblock

    dfm2['scoress_bin'] = dfm2[col_score].apply(lambda x: np.floor(x / 1.0) * 1.0)
    dfs5 = dfm2.groupby('scoress_bin').agg({col_score: 'mean',
                                            coldefault: {'sum', 'count'}
                                            }).reset_index()
    dfs5.columns = [x[0] if x[0] == x[1] else x[0] + '_' + x[1] for x in dfs5.columns]
    dfs5 = dfs5.sort_values(col_score, ascending=False)
    # return dfs5

    l2 = []
    k = 1
    ndef, nuser = 0, 0
    for i, x in dfs5.iterrows():
        if k > nblock: break
        nuser = nuser + x[coldefault + '_count']
        ndef = ndef + x[coldefault + '_sum']
        pdi = ndef / nuser

        if ndef > def_list[k - 1]:
            # if  pdi > pdlist[k] :
            l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
            k = k + 1
            ndef, nuser = 0, 0
    l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
    l2 = pd.DataFrame(l2, columns=[col_score, 'kaiso3', 'pd', 'ndef', 'nuser'])
    return l2


##### Get Kaiso limit ###############################################################
def fun_get_segmentlimit(x, l1):
    for i in range(0, len(l1)):
        if x >= l1[i]:
            return i + 1
    return i + 1 + 1


def np_drop_duplicates(l1):
    l0 = list(OrderedDict((x, True) for x in l1).keys())
    return l0


def col_extractname_colbin(cols2):
    coln = []
    for ss in cols2:
        xr = ss[ss.rfind("_") + 1:]
        xl = ss[:ss.rfind("_")]
        if len(xr) < 3:  # -1 or 1
            coln.append(xl)
        else:
            coln.append(ss)

    coln = np_drop_duplicates(coln)
    return coln


def pd_stat_col(df):
    ll = {'col': [], 'nunique': []}
    for x in df.columns:
        ll['col'].append(x)
        ll['nunique'].append(df[x].nunique())
    ll = pd.DataFrame(ll)
    n = len(df) + 0.0
    ll['ratio'] = ll['nunique'] / n
    ll['coltype'] = ll['nunique'].apply(lambda x: 'cat' if x < 100 else 'num')

    return ll


def pd_col_intersection(df1, df2, colid):
    n2 = list(set(df1[colid].values).intersection(df2[colid]))
    print("total matchin", len(n2), len(df1), len(df2))
    return n2


def pd_col_normalize(dfm2, colnum_log, colproba):
    for x in ['SP1b', 'SP2b']:
        dfm2[x] = dfm2[x] * 0.01

    dfm2['SP1b'] = dfm2['SP1b'].fillna(0.5)
    dfm2['SP2b'] = dfm2['SP2b'].fillna(0.5)

    for x in colnum_log:
        try:
            dfm2[x] = np.log(dfm2[x].values.astype(np.float64) + 1.1)
            dfm2[x] = dfm2[x].replace(-np.inf, 0)
            dfm2[x] = dfm2[x].fillna(0)
            print(x, dfm2[x].min(), dfm2[x].max())
            dfm2[x] = dfm2[x] / dfm2[x].max()
        except:
            pass

    for x in colproba:
        print(x)
        dfm2[x] = dfm2[x].replace(-1, 0.5)
        dfm2[x] = dfm2[x].fillna(0.5)

    return dfm2


def pd_col_check(dfm2):
    for x in dfm2.columns:
        if len(dfm2[x].unique()) > 2 and dfm2[x].dtype != np.dtype('O'):
            print(x, len(dfm2[x].unique()), dfm2[x].min(), dfm2[x].max())


def pd_col_remove(df, cols):
    for x in cols:
        try:
            del df[x]
        except:
            pass
    return df







def col_extractname(col_onehot):
    '''
    Column extraction 
    '''
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
    '''
    colsremove = [
    'y', 'def',
    'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
    'score' ,   'segment2' ,
    'scoreb', 'score_kaisob', 'segmentb', 'def_test'
    ]
    colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    '''
    for x in colsremove:
        try:
            cols.remove(x)
        except:
            pass
    return cols


def col_remove_fuzzy(cols, colsremove):
    # cols = list(df1.columns)
    '''
      colsremove = [
         'y', 'def',
         'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
         'score' ,   'segment2' ,
         'scoreb', 'score_kaisob', 'segmentb', 'def_test'
      ]
      colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
    '''
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
    df1 = copy.deepcopy(dfxx[cols + ['def', 'y']])
    df1 = df1[df1['def'] < 201905]
    df1 = df1[(df1['def'] > 201703) | (df1['def'] == -1)]
    return df1


def col_feature_importance(Xcol, Ytarget):
    """ random forest for column importance """
    pass


def col_study_getcategorydict_freq(catedict):
    """ Generate Frequency of category : Id, Freq, Freqin%, CumSum%, ZScore
      given a dictionnary of category parsed previously
  """
    catlist = []
    for key, v in list(catedict.items()):
        df = util.pd_array_todataframe(util.np_dict_tolist(v), ["category", "freq"])
        df["freq_pct"] = 100.0 * df["freq"] / df["freq"].sum()
        df["freq_zscore"] = df["freq"] / df["freq"].std()
        df = df.sort_values(by=["freq"], ascending=0)
        df["freq_cumpct"] = 100.0 * df["freq_pct"].cumsum() / df["freq_pct"].sum()
        df["rank"] = np.arange(0, len(df.index.values))
        catlist.append((key, df))
    return catlist


def col_pair_correl(Xcol, Ytarget):
    pass


def col_pair_interaction(Xcol, Ytarget):
    """ random forest for pairwise interaction """
    pass


def col_study_summary(Xmat=[0.0, 0.0], Xcolname=["col1", "col2"], Xcolselect=[9, 9], isprint=0):
    n, m = np.shape(Xmat)
    if Xcolselect == [9, 9]:
        Xcolselect = np.arange(0, m)
    if len(Xcolname) != m:
        print("Error column size: ")
        return None
    colanalysis = []
    for icol in Xcolselect:
        Xraw_1unique = np.unique(Xmat[:, icol])
        vv = [
            Xcolname[icol],
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
        print(("Nb_Samples:", np.shape(Xmat)[0], "Nb Col:", len(Xcolname)))
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


def pd_na_missing_show():
    """
   https://blog.modeanalytics.com/python-data-visualization-libraries/


   Missing Data

     missingno
import missingno as msno
%matplotlib inline
msno.matrix(collisions.sample(250))
At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be completely populated, while geographic information seems mostly complete, but spottier.

The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum rows.

This visualization will comfortably accommodate up to 50 labelled variables. Past that range labels begin to overlap or become unreadable, and by default large displays omit them.


Heatmap
The missingno correlation heatmap lets you measure how strongly the presence of one variable positively or negatively affect the presence of another:
msno.heatmap(collisions)


https://github.com/ResidentMario/missingno

   """


def pd_stat_describe(df):
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
def pd_cat_label_toint(Xmat):
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
    le = sk.preprocessing.LabelEncoder()
    ncol = Xmat.shape[1]
    Xnew = np.zeros_like(Xmat)
    mapping_cat_int = {}

    for k in range(0, ncol):
        Xnew[:, k] = le.fit_transform(Xmat[:, k])
        mapping_cat_int[k] = le.get_params()

    return Xnew, mapping_cat_int




