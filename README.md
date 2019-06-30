# da

```
Naming convention for functions, arguments :


## Function naming
pd_   :  input is pandas dataframe
np_ : input is numpy
sk_  :  inout is related to sklearn (ie sklearn model)

_col_  :  name for colums
_colcat_  :  name for category columns
_colnum_  :  name for numerical columns

col_ :  function name for column list related.



### Variables naming  ############################################
df     :  variable name for dataframe
colcat : For category column
colnum :  For numerical columns
coldate : for date columns



##################################################################
For module file has LIMITED dependency and logic flow :
   util_feature.py  :  Input/Outout should be pandas, pandas-like
   util_model.py :  Input/Output should be numpy



```











