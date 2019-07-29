"""
Some special methods

"""

try:
    from geopy.distance import great_circle
except Exception as e:
    print(e)


# ---------------------Execute rules on State Matrix --------------------------------
class sk_stateRule:
    """ Calculate Rule(True/False) based on State and Trigger
      Allow to add function to class externally                     """

    def __init__(self, state, trigger, colname=[]):
        self.lrule = np.empty((3, 20), dtype=np.object)
        self.nrule = 20
        sh = np.shape(state)
        self.tmax = sh[0]
        self.nstate = sh[1]
        sh = np.shape(trigger)
        self.ktrigger = sh[0]
        self.ntrigger = sh[1]

        if len(colname) > 1:
            self.colname = colname
        else:
            self.colname = ["a" + str(i) for i in range(0, self.nstate)]

        self.state = util.np_torecarray(state, self.colname)
        self.trigger = util.np_torecarray(trigger, self.colname)

    def addrule(self, rulefun, name="", desc=""):
        kid = util.findnone(self.lrule[0, :])
        kid2 = util.find(name, self.lrule[1, :])
        if kid2 != -1 and name != "":
            print("Name already exist !")
        else:
            if kid == -1:
                lrule = util.np_addcolumn(self.lrule, 50)
                kid = self.nrule

            try:
                test = rulefun(self.state, self.trigger, 1)
                self.lrule[0, kid] = copy.deepcopy(rulefun)
                self.lrule[1, kid] = name
                self.lrule[2, kid] = desc
            except ValueError as e:
                print(("Error with the function" + str(e)))

    def eval(self, idrule, t, ktrig=0):
        if isinstance(idrule, str):  # Evaluate by name
            kid = util.find(idrule, self.lrule[1, :])
            if kid != -1:
                return self.lrule[0, kid](self.state, self.trigger, t)
            else:
                print(("cannot find " + idrule))
        else:
            return self.lrule[0, idrule](self.state, self.trigger, t)

    def help(self):
        """
s1= np.arange(5000).reshape((1000, 5))
trig1= np.ones((1,5))
state1= sk_stateRule(aa, trig1, ['drawdown','ma100d','ret10d','state_1','state_2'] )

def fun1(s, tr,t):
  return  s.drawdown[t] < tr.drawdown[0] and  s.drawdown[t] < tr.drawdown[0]

def fun2(s, tr,t):
 return  s.drawdown[t] > tr.drawdown[0] and  s.drawdown[t] < tr.drawdown[0]

state1.addrule(fun1, 'rule6')
state1.addrule(fun2, 'rule5')

state1.eval(idrule=0,t=5)

state1.eval(idrule=1,t=5)

state1.eval(idrule='rule5',t=6)

util.save_obj(state1, 'state1')

np.shape(aa2)

aa2= util.np_torecarray(aa,  ['drawdown','a2','a3','a4','a5'])

util.find(5.0, aa2[0])

recordarr = np.rec.array([(1,2.,7),(2,3.,5)],
                   dtype=[('col1', 'f8'),('col2', 'f8'), ('col3', 'f8')])
recordarr.col3[0]

state1= stateRule(np.ones((100,10)), np.ones((1,10)))

col= aa2.a2

"""


"""

def (X):
    return X[:, 1:]

def drop_first_component(X, y):
    "" Create a pipeline with PCA and the column selector and use it to transform the dataset. ""
    pipeline = make_pipeline( PCA(), FunctionTransformer(all_but_first_column))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    return pipeline.transform(X_test), y_test


"""


def optim_de(obj_fun, bounds, maxiter=1, name1="", solver1=None, isreset=1, popsize=15):
    """ Optimization and Save Data into file"""
    import copy

    if isreset == 2:
        print("Traditionnal Optim, no saving")
        res = sci.optimize.differential_evolution(obj_fun, bounds=bounds, maxiter=maxiter)
        xbest, fbest, solver, i = res.x, res.fun, "", maxiter
    else:  # iterative solver
        print("Iterative Solver ")
        if name1 != "":  # wtih file
            print("/batch/" + name1)
            solver2 = load_obj("/batch/" + name1)
            imin = int(name1[-3:]) + 1
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            solver.population = copy.deepcopy(solver2.population)
            solver.population_energies = copy.deepcopy(solver2.population_energies)
            del solver2

        elif solver1 is not None:  # Start from zero
            solver = copy.deepcopy(solver1)
            imin = 0
        else:
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            imin = 0

        name1 = "/batch/solver_" + name1
        fbest0 = 1500000.0
        for i in range(imin, imin + maxiter):
            xbest, fbest = next(solver)
            print(0, i, fbest, xbest)
            res = (copy.deepcopy(solver), i, xbest, fbest)
            try:
                util.save_obj(solver, name1 + util.date_now() + "_" + util.np_int_tostr(i))
                print((name1 + util.date_now() + "_" + util.np_int_tostr(i)))
            except:
                pass
            if np.mod(i + 1, 11) == 0:
                if np.abs(fbest - fbest0) < 0.001:
                    break
                fbest0 = fbest

    return fbest, xbest, solver


######################### OPTIM   ###################################################
def optim_is_pareto_efficient(Xmat_cost, epsilon=0.01, ret_boolean=1):
    """ Calculate Pareto Frontier of Multi-criteria Optimization program
    c1, c2  has to be minimized : -Sharpe, -Perf, +Drawdown
    :param Xmat_cost: An (n_points, k_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    pesp = 1.0 + epsilon  # Relax Pareto Constraints
    is_efficient = np.ones(Xmat_cost.shape[0], dtype=bool)
    for i, c in enumerate(Xmat_cost):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                Xmat_cost[is_efficient] <= c * pesp, axis=1
            )  # Remove dominated points
    if ret_boolean:
        return is_efficient
    else:
        return Xmat_cost[is_efficient]
    # return is_efficient


def pd_validation_struct():
    pass
    """
  https://github.com/jnmclarty/validada

  https://github.com/ResidentMario/checkpoints


  """


def pd_checkpoint():
    pass


########### Added functions
##################################


def pd_col_add_distance_to_point(df, center_point):
    """
    Function adding a column with distance to a point
    Arguments:
        df:            dataframe (requires latitude / longitude)
        center_point:  tuple with lat / long of point
    Returns:
        df:            dataframe with new column ['distance'] (in km)
    """
    if "latitude" and "longitude" not in df.columns:
        print("There are no lat / long data in the dataframe")
    else:
        df["distance"] = df.apply(
            lambda x: round(great_circle(center_point, (x.latitude, x.longitude)).km, 3), axis=1
        )
    return df


def np_correl_rank(correl=[[1, 0], [0, 1]]):
    """ Correl Ranking:  Col i, Col j, Correl_i_j, Abs_Correl_i_j    """
    m, n = np.shape(correl)
    correl_rank = np.zeros((n * (n - 1) / 2, 3), dtype=np.float32)
    k = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            k += 1
            correl_rank[k, 0] = i
            correl_rank[k, 1] = j
            correl_rank[k, 2] = correl[i, j]
            correl_rank[k, 3] = abs(correl[i, j])
    correl_rank = util.sortcol(correl_rank, 3, asc=False)
    return correl_rank


def sk_cluster_distance_pair(Xmat, metric="jaccard"):
    """
    'euclidean, 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean, 'cosine, 'correlation, 'hamming, 'jaccard, 'chebyshev, 'canberra, 'braycurtis, 'mahalanobis', VI=None) 'yule, 'matching, 'dice, 'kulsinski, 'rogerstanimoto, 'russellrao, 'sokalmichener, 'sokalsneath,

    'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}
   #Visualize discretization scheme

   Xtrain_dist= sci.spatial.distance.squareform(sci.spatial.distance.pdist(Xtrain_d,
             metric='cityblock', p=2, w=None, V=None, VI=None))

   Xtsne= da.plot_cluster_tsne(Xtrain_dist, metric='', perplexity=40, ncomponent=2, isprecompute=True)

   """
    import fast

    if metric == "jaccard":
        return fast.distance_jaccard_X(Xmat)

    else:  # if metric=='euclidian'
        return sci.spatial.distance.squareform(
            sci.spatial.distance.pdist(Xmat, metric=metric, p=2, w=None, V=None, VI=None)
        )










######################  Category Classifier Trees  #########################################################################
"""
Category Classifier
https://github.com/catboost/catboost/blob/master/catboost/tutorials/kaggle_paribas.ipynb

Very Efficient
D:\_devs\Python01\project27\linux_project27\mlearning\category_learning


https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/


clf = CatBoostClassifier(learning_rate=0.1, iterations=1000, random_seed=0)
clf.fit(train_df, labels, cat_features=cat_features_ids)


##### Base Approach
import pandas as pd
import numpy as np

from itertools import combinations
from catboost import CatBoostClassifier


labels = train_df.target
test_id = test_df.ID

train_df.drop(['ID', 'target'], axis=1, inplace=True)
test_df.drop(['ID'], axis=1, inplace=True)

train_df.fillna(-9999, inplace=True)
test_df.fillna(-9999, inplace=True)

# Keep list of all categorical features in dataset to specify this for CatBoost
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()



########  Regularizer
selected_features = [
    'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50',
    'v52', 'v56', 'v62', 'v66', 'v72', 'v75', 'v79', 'v91', 'v112', 'v113', 'v114', 'v129'
]

# drop some of the features that were not selected
train_df = train_df[selected_features]
test_df = test_df[selected_features]

# update the list of categorical features
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()


char_features = list(train_df.columns[train_df.dtypes == np.object])
char_features_without_v22 = list(train_df.columns[(train_df.dtypes == np.object) & (train_df.columns != 'v22')])

cmbs = list(combinations(char_features, 2)) + map(lambda x: ("v22",) + x, combinations(char_features_without_v22, 2))


clf = CatBoostClassifier(learning_rate=0.1, iterations=1000, random_seed=0)
clf.fit(train_df, labels, cat_features=cat_features_ids)


"""





# ---------------------             ----------------
"""
 Reshape your data either using X.reshape(-1, 1) if your data has a single feature or
  X.reshape(1, -1) if it contains a single sample.

"""


"""
  Create Checkpoint on dataframe to save intermediate results
  https://github.com/ResidentMario/checkpoints
  To start, import checkpoints and enable it:

from checkpoints import checkpoints
checkpoints.enable()
This will augment your environment with pandas.Series.safe_map and pandas.DataFrame.safe_apply methods. Now suppose we create a Series of floats, except for one invalid entry smack in the middle:

import pandas as pd; import numpy as np
rand = pd.Series(np.random.random(100))
rand[50] = "____"
Suppose we want to remean this data. If we apply a naive map:

rand.map(lambda v: v - 0.5)

    TypeError: unsupported operand type(s) for -: 'str' and 'float'
Not only are the results up to that point lost, but we're also not actually told where the failure occurs! Using safe_map instead:

rand.safe_map(lambda v: v - 0.5)

    <ROOT>/checkpoint/checkpoints/checkpoints.py:96: UserWarning: Failure on index 50
    TypeError: unsupported operand type(s) for -: 'str' and 'float'


"""


"""
You can control how many decimal points of precision to display
In [11]:
pd.set_option('precision',2)

pd.set_option('float_format', '{:.2f}'.format)


Qtopian has a useful plugin called qgrid - https://github.com/quantopian/qgrid
Import it and install it.
In [19]:
import qgrid
qgrid.nbinstall()
Showing the data is straighforward.
In [22]:
qgrid.show_grid(SALES, remote_js=True)


SALES.groupby('name')['quantity'].sum().plot(kind="bar")


"""


"""
class META_DB_CLASS(object):
   # Create Meta database to store infos on the tables : csv, zip, HFS, Postgres
ALL_DB['japancoupon']= {}
ALL_DB['japancoupon']['schema']=    df_schema
ALL_DB['japancoupon']['table_uri']= df_schema
ALL_DB['japancoupon']['table_columns']= df_schema


   def __init__(self, db_file='ALL_DB_META.pkl') :
     if db_file.find('.pkl') != -1 :
      self.filename= db_file
      self.db= util.load(db_file, isabsolutpath=1)

   def db_add(self, dbname ):
     self.db[dbname]= {}    # util.np_dictordered_create()

   def db_update_item(self, dbname, itemlistname='table_uri/schema/table_columns', itemlist=[]):
     self.db[dbname][itemlistname]=  itemlist

   def db_save(self, filename='') :
     if filename== '' :
        util.save(self.db, self.filename, isabsolutpath=1)
     else :
        self.filename= filename
        util.save(self.filename)

   def db_print_item(self):
       pass

meta_db= META_DB_CLASS( in1+'ALL_DB_META.pkl')

"""


############################################################################
# ---------------------             --------------------
"""
Symbolic Regression:

http://gplearn.readthedocs.io/en/latest/examples.html#example-2-symbolic-tranformer


!pip install gplearn


x0 = np.arange(-1, 1, 1/10.)
x1 = np.arange(-1, 1, 1/10.)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                       color='green', alpha=0.5)
plt.show()

import gplearn as gp

rng = gp.check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

"""

# ------¨Pre-Processors --------------------------------------------------------------
"""
One-Hot: one column per category, with a 1 or 0 in each cell for if the row contained that column’s category
Binary: first the categories are encoded as ordinal, then those integers are converted into binary code,
then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot,
 but with some distortion of the distances.

http://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html

import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...])
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])

Best is Binary Encoder

Splice
Coding	Dimensionality	Avg. Score	Elapsed Time
14	Ordinal	61	0.68	5.11
17	Sum Coding	3465	0.92	25.90
16	Binary Encoded	134	0.94	3.35
15	One-Hot Encoded	3465	0.95	2.56


Value ---> Hash  (limited in value)
      ---> Reduce Dimensionality of the Hash

def hash_fn(x):
tmp = [0for_inrange(N)]
for val in x.values:
tmp[hash(val)% N] += 1
return pd.Series(tmp, index=cols)

cols = ['col_%d'% d for d in range(N)]
X = X.apply(hash_fn, axis=1)


@profile(precision=4)
def onehot():
    X, _, _ = get_mushroom_data()
    print(X.info())
    enc = ce.OneHotEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out

def binary(X):
    enc = ce.BinaryEncoder()
    enc.fit(X, None)
    out = enc.transform(X)
    print(out.info())
    del enc, _, X, out

enc = ce.OneHotEncoder()
X_bin = enc.fit_transform(X)

import matplotlib.pyplot as plt
import category_encoders as ce
from examples.source_data.loaders import get_mushroom_data, get_cars_data, get_splice_data


"""
