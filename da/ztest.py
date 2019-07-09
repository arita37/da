# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload 2


from util_feature import *




##########################################################################
df = pd.read_csv( "data/titanic_train.csv")


df1 = pd_colnum_tocat_quantile( df, colname=[ "Fare" ],   bins=5,
                          suffix="_bin" )

df[ "Fare_bin" ] 












"""
df[ "Fare_bin" ] = df[ "Fare_bin" ].astype("int")
df[[ "Fare_bin" ]].hist()

df[[ "Fare_bin", "Fare" ]]









c = "Fare"


"""









