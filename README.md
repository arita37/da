# da

```
Naming convention for functions, arguments :

##################################################################
For module file has LIMITED dependency and logic flow :
   util_feature.py  :  Input/Outout should be pandas, pandas-like
   util_model.py :  Input/Output should be numpy
   util_plot.py   :  Input * mostly numpy or pandas



## Function naming   #################################################
pd_   :  input is pandas dataframe
np_ : input is numpy
sk_  :  inout is related to sklearn (ie sklearn model)
plot_


_col_  :  name for colums
_colcat_  :  name for category columns
_colnum_  :  name for numerical columns


_stat_ : show statistics
_df_  : dataframe
_num_ : statistics

col_ :  function name for column list related.



### Variables naming  ############################################
df     :  variable name for dataframe
colname
colexclude
colcat : For category column
colnum :  For numerical columns
coldate : for date columns



#########
Auto formatting
isort -rc .
black --line-length 100




#########
conda create -n py36_tf13 python=3.6.5  -y
source activate py36_tf13
conda install  -c anaconda  tensorflow=1.13.1
conda install -c anaconda scikit-learn pandas matplotlib seaborn -y
conda install -c anaconda  ipykernel -y



```











