# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload 2


from util_feature import *




##########################################################################
df = pd.read_csv( "data/titanic_train.csv")


df1 = pd_colnum_tocat_quantile( df, colname=[ "Fare" ],   bins=5,
                          suffix="_bin" )

df[ "Fare_bin" ] 




def pd_col_findtype(df) :
  """
  :param df:
  :return:
  """
  n = len(df) + 0.0
  colcat , colnum, coldate, colother = [], [], [], []
  for x in df.columns :
      nunique = len( df[x].unique())
      ntype = str(df[x].dtype)
      r =  nunique /n
      print(r, nunique, ntype )


      if r > 0.90 :
          colother.append(x)


      elif nunique < 3 :
          colcat.append(x)

      elif ntype == "o"  :
          colcat.append(x)

      elif nunique > 50 and ( "float" in ntype or  "int" in ntype ):
          colnum.append(x)

      else :
          colother.append(x)

  return colcat , colnum, coldate, colother



pd_col_findtype(df) 













"""
df[ "Fare_bin" ] = df[ "Fare_bin" ].astype("int")
df[[ "Fare_bin" ]].hist()

df[[ "Fare_bin", "Fare" ]]









c = "Fare"


"""









