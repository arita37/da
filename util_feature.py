"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas

"""

def tohot(df, colnames):
    pass


def one_hot_encoder(df, nan_as_category = True, categorical_columns=None):
    pass


def num_tocat(df):
    pass
   
def feature_impt_logis(clf, cols2) :
    pass

def merge_columns( dfm3, ll0 ) :
    pass


def merge_colunns2( dfm3,  l, x0 ) :
    pass


def downsample(df, coltarget="y", n1max= 10000, n2max= -1, isconcat=1 ):
    pass
   """
      DownSampler      
   """

# return list(df.columns)
def get_ccol(df):
    pass


def get_dfna_col(dfm2):
    pass


def get_dfna_user(dfm2, n = 10**6):
    pass


def get_stat_imbalance(df):
    pass



def get_cat_correlation_ratio(categories, measurements):
    pass


# ???
def theils_u(x, y):
    pass    
    
    
#### Calculate KAISO Limit  #########################################################
def get_kaiso_limit(dfm2, col_score='scoress', coldefault="y", ntotal_default=491, def_list=None, nblock=20.0) : 
    pass


##### Get Kaiso limit ###############################################################
def get_kaiso2(x, l1):
    pass

##### Drop duplicates
def drop_duplicates(l1):
    pass

def col_extract_colbin(cols2) :
    pass


def col_stats(df) :
    pass


def intersection(df1, df2, colid) :
    pass


def feat_normalize(dfm2, colnum_log, colproba) :
    pass


def feat_check( dfm2 ) :
    pass


def remove(df, cols) :
    pass


def feature_importance(clf, cols) :
    pass


def col_extract(colbin):
    pass
 '''
    Column extraction 
 '''   

def col_remove(cols, colsremove) :
    pass


def col_remove_fuzzy(cols, colsremove) :
    pass




def feature_filter(dfxx , cols ) :      
    pass


def col_feature_importance(Xcol, Ytarget):
    """ random forest for column importance """
    pass

def col_study_getcategorydict_freq(catedict):
    pass
    """ Generate Frequency of category : Id, Freq, Freqin%, CumSum%, ZScore
      given a dictionnary of category parsed previously
  """
      
def col_pair_correl(Xcol, Ytarget):
    pass


def col_pair_interaction(Xcol, Ytarget):
    """ random forest for pairwise interaction """
    pass


def col_study_summary(Xmat=[0.0, 0.0], Xcolname=["col1", "col2"], Xcolselect=[9, 9], isprint=0):
    pass

def filter_column(df_client_product, filter_val=[], iscol=1):
    pass
    """
   # Remove Columns where Index Value is not in the filter_value
   # filter1= X_client['client_id'].values
   :param df_client_product:
   :param filter_val:
   :param iscol:
   :return:
   """

def missing_show():
    pass

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


def describe(df):
    """ Describe the tables


   """
    pass

def stack_dflist(df_list):
    pass


######################  Transformation   ###########################################################

def transform_catlabel_toint(Xmat):
    
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
    pass

def transform_pca(Xmat, dimpca=2, whiten=True):
    pass
    """Project ndim data into dimpca sub-space  """


######################### OPTIM   ###################################################
def optim_is_pareto_efficient(Xmat_cost, epsilon=0.01, ret_boolean=1):
    """ Calculate Pareto Frontier of Multi-criteria Optimization program
    c1, c2  has to be minimized : -Sharpe, -Perf, +Drawdown
    :param Xmat_cost: An (n_points, k_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    pass






