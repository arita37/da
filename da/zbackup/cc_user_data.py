# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload
import os, sys
DIRCWD=  'G:/_devs/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage'); # sys.path.append(DIRCWD + '/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
import util,  numpy as np, gc

import  pandas as pd, sqlalchemy as sql, dask.dataframe as dd, dask, datanalysis as da, arrow
from attrdict import AttrDict as dict2
from collections import defaultdict
############################################################################################


######################### FOLDERS ##########################################################
execfile( DIRCWD + '/aapackage/coke_functions.py')
execfile( DIRCWD + '/unerry/cc_folders.py')



################## Data Type, File #########################################################
# df= df.astype(dtype0)
dtype0= {'amount': 'int16',    'area_code': 'int32',         'bottler_code': 'int16',
 'bottler_name': 'category',   'id': 'int32',
 'js_code': 'int32',           'location_code': 'int32',
 'machine_code': 'int64',      'price': 'int16',
 'product_name': 'category',   'promotion_id': 'float16',
 'purchased_at': 'object',     'temperature': 'int16',  'user_id': 'int32',
 'month': 'int8',  'day': 'int8'   
}

gc.collect()




############################################################################################ 
############################################################################################
#### 1) Read CSV FULL: partial columns,   ##################################################
#          0          1                2                 3                4
usecols=["user_id", "machine_code", "location_code",  "bottler_code", 'product_name', 
         'temperature', "amount",'purchased_at', 'user_rank_vol']
'''



   
   
   


'''





G:\_data\raw\cokeon_csv\201704


df= pd.read_csv(uri.rawcokeon+'/201704/purchasings_big.csv', dtype= dtype0, nrows=10**8)       


df= pd.read_csv(uri.rawcokeon+'/purchasings.csv', dtype= dtype0, nrows=10**8)       


df= pd.read_csv(uri.rawcokeon+'/purchasings_20170206_BIG.csv', dtype= dtype0, nrows=10**8)       


########### Basho Area  ####################################################################
df_fsmaster= pd.read_csv( uri.map + '/map_fsmaster_unerry.txt').fillna('0')
df_fsmaster.columns
''' Index([u'machine_code', u'cokeonarea', u'customer_name', u'basho_id',
       u'basho_name', u'inout', u'basho_type', u'Business condition code',
       u'basho_type_name', u'address', u'address_nb', u'Unnamed: 11',
       u'Unnamed: 12', u'Unnamed: 13', u'Unnamed: 14', u'Unnamed: 15'],
'''
df_fsmaster["machine_code"]=  df_fsmaster["machine_code"].astype('int64') 
df_fsmaster["basho_type2"]=   df_fsmaster["basho_type"].apply( lambda x : x[0:2] )




map_basho=     util.pd_df_todict(df_fsmaster, colkey= "machine_code" , colval= "basho_type2" )  
def add_bashotype(x) :
   try :      return map_basho[x]
   except :   return '' 
df1["bashotype"]= df1["machine_code"].apply(add_bashotype).astype('category') 



map_basho_id=     util.pd_df_todict(df_fsmaster, colkey= "machine_code" , colval= "basho_id" )  
def add_basho_id(x) :
   try :      return int(map_basho_id[x])
   except :   return '' 
df1["basho_id"]= df1["machine_code"].apply(add_basho_id).astype('category') 


############################################################################################






############################################################################################
###### Soromasu files ######################################################################
#  G:\_data\raw\cokeon_csv\20170630
### Load all files into single
'''
No.,日時,エリアコード,ボトラー名,ボトラーコード,マシンコード,ロケーションコード,JS商品CD,JS商品名,購買価格,HOT/COLD,スタンプボーナスID,サンプリング識別ID,スタンプ付与数,リワード付与数,リワードチケット利用数,サンプリング利用数,ユーザーID,自販機ID,チケットシリアル,製品ID,決済種別,決済コード,コンテンツプロモーションID
'''  


list_files= util.os_file_listall( uri.raw + r'/cokeon_csv/20170630/', r'*.csv', 1)


dfi0= None
cols= ["No.", "date", "area code", "bottler_name", "bottler_code", "machine_code", "location_code", 
"js_code", "js_product_name", "purchase_price", "HOT_COLD", "stampbonusID", "samplingID", 
"stampnumber", "rewardnumber" , "Number of reward tickets used", "number of sampling use", 
"user_id", "machine_code_ID", "ticket_serial", "product ID", "settlement type", "payment code", 
"content promotion ID" ]


for f in list_files[2] :
  print f
  dfi= pd.read_csv(f , dtype= dtype0, nrows=10**8  )
  dfi.columns= cols
  
  try :    dfi0= pd.concat((dfi0, dfi))    
  except : dfi0= dfi       


dfi0.sort_values("date", ascending=1, inplace= True)
dfi0.dtypes


# dfi0.to_csv( r'G:/_data/raw/cokeon_csv/CCW_0523/OK/cokeon_ccw_0301_0701.csv', index=False, mode='w')
# df.columns





############################################################################################
############################################################################################
dfi0= pd.read_csv( r'G:/_data/raw/cokeon_csv/CCW_0523/OK/cokeon_ccw_0421_0604.csv')


# スタンプ付与数 : amount in nb of stamps
dfi0= dfi0[[ "date",  "machine_code", "js_code",  "stampnumber",
             "js_product_name", "purchase_price", "HOT_COLD", "user_id"]]


dfi0.columns= [ "purchased_at",  "machine_code", "js_code",  "amount",
                "product_name", "price", "temperature", "user_id"]

dfi0["basho_id"]= dfi0["machine_code"].apply(add_basho_id)


def add_dateday(x) :
  try :   
     a= arrow.get(x, "YYYY/M/D" )
     return 10000* a.year + 100*a.month + a.day
  except : 
     return 0


dfi0['amount']=   dfi0['amount'].apply( lambda x : 1 if x==0 else x  )     
dfi0['dateday']=  dfi0['purchased_at'].apply( add_dateday  )

dfi0["month"]= dfi0['dateday'].apply(  lambda x :  str(x)[4:6]  )

dfi0["year"]= dfi0['dateday'].apply(   lambda x :  str(x)[0:4]  )
dfi0.sort_values("dateday", ascending=1, inplace=1)

dfi0["bashotype"]= dfi0["machine_code"].apply(add_bashotype).astype('category') 




############################################################################################
####             ###########################################################################
dfi0.dtypes
print dfi0.head()

df11= dfi.head()

len(   dfi0[ -( dfi0["number of sampling use"]>0)   ] )


dfi0=  dfi0[ -( dfi0["number of sampling use"]>0 )   ] 






            
            
            
            


            
            
            
            




############################################################################################
############################################################################################
'''
535553


No.                                int64
date                              object
area code                          int64


bottler_name                      object
bottler_code                       int64
machine_code                       int64
location_code                      int64


js_code                          float64
js_product_name                   object
purchase_price                     int64
HOT_COLD                          object


stampbonusID                     float64
samplingID                       float64
stampnumber                        int64
rewardnumber                      object
Number of reward tickets used     object
number of sampling use            object


user_id                            int64
machine_code                       int64
ticket_serial                    float64
product ID                         int64
settlement type                    int64
payment code                      object
content promotion ID              object



'''








############################################################################################
dfi0.to_csv( r'G:/_data/raw/cokeon_csv/cokeon_CCW_201603_201707_big1.csv', index=0, mode='w')


#####      #####
min( dfi0.dateday.values  ), max( dfi0.dateday.values  )

df1=  dfi0
############################################################################################
############################################################################################



dfa= pd.read_csv( uri.raw + '/cokeon_csv/20170201/purchasings_20170206_BIG.csv' )
dfa.dtypes


dfa.describe()




#####################################################################################
#######                 #############################################################
def add_date(x):
   return  x["year"] * 10000    +    x["month"] * 100     +    x["day"] 

dfa["date"]= dfa.apply(add_date, axis=1)



def pervm(dfi) :
   dff= dfi.sort_values("date", ascending=1)
   return pd.DataFrame( [dff.date.values[0]], columns=["date"])

dfa1= dfa.groupby("machine_code").apply(pervm)

############################################################################################
############################################################################################




############################################################################################
##########             #####################################################################
dfa["mindate"]

df1= dfa[ ( dfa.is_ccwj==1 ) & ( dfa.mindate < 20160702)  ]

len( df1.machine_code.unique() )


## 13336




############################################################################################
df2b.columns
df2b.dtypes

'''
purchased_at      object
machine_code       int64
js_code          float64
amount             int64
product_name      object
price              int64
temperature       object
user_id            int64
basho_id          object
dateday            int64
month             object
year              object
bashotype       category
h0_group        category
test_group      category
yearmonth          int64

'''



############################################################################################
############################################################################################
set(  df.columns  ) == set(  dfi0.columns )


dfi1=  dfi0[ dfi0.dateday > 20170423 ]  


len(dfi1)


aa= dfi0.purchased_at.unique()

aa[0].split("/")


arrow.get(aa[0], "YYYY/M/D" ).day


len(dfi0.dateday.unique())


x= aa[0]
10000 *int(x[0:4]) +  100 * int(x[5:6]) +  int(x[7:8] )




############################################################################################
############################################################################################
dfi0.head(20)


max(df.dateday)


df1= pd.concat((df, dfi1))


x= '2017/04/11 00:00:05'

df1.to_csv( r'G:/_data/raw/cokeon_csv/cokeon_CCW_201604_201705_big1.csv', index=0, mode='w')
############################################################################################
############################################################################################


max(df.dateday)
20170423


min(dfi1.dateday)
20170501






############################################################################################
###########  H0 VM list ####################################################################
df_fsh0= pd.read_csv(  r'G:/_data/raw/cokeon_csv/h0_vm_list.txt' ).fillna('0')
df_fsh0.columns
'''
Index([u'machine_code', u'basho_code', u'H0_Group', u'TESTGROUP', u'営業本部ｺｰﾄﾞ',
       u'営業本部', u'エリア', u'No.', u'商流営業ｺｰﾄﾞ', u'商流支店名', u'商流ﾙｰﾄ', u'物流事業所ｺｰﾄﾞ',
       u'物流事業所名', u'物流ﾙｰﾄ', u'得意先ｺｰﾄﾞ', u'得意先名称', u'納品場所ｺｰﾄﾞ', u'納品場所名',
       u'マシンコード', u'basho_type', u'戦略ﾁｬﾈﾙ', u'機種ｺｰﾄﾞ', u'機種名称', u'年式',
       u'nb_columns', u'nb_pet', u'対象外', u'フラグ', u'年間見込み', u'セールスランク'],
'''

map_h0=  util.pd_df_todict(df_fsh0, colkey= "machine_code" ,   colval= "H0_Group" )  

def add_map_h0(x) :
   try :      return map_h0[x]
   except :   return ''
df1["h0_group"]= df1["machine_code"].apply(add_map_h0).astype('category') 


map_test=  util.pd_df_todict(df_fsh0, colkey= "machine_code" , colval= 'TESTGROUP' )  
def add_map_test(x) :
   try :      return map_test[x]
   except :   return ''
df1["test_group"]= df1["machine_code"].apply(add_map_test).astype('category')    


df1["yearmonth"]= df1["dateday"].apply( lambda x : int(str(x)[0:6]))


len(  df1[df1.test_group != '' ].machine_code.unique()  )
############################################################################################





############################################################################################
########### per VM #########################################################################
df_pvt= pd.pivot_table(df1, index= [ "machine_code"], columns=["yearmonth"],  
                       values= 'user_id' ,   aggfunc=pd.Series.nunique ).fillna(0).reset_index()  


df_pvt["test_group"]= df_pvt["machine_code"].apply(add_map_test)    
df_pvt["bashotype"]=  df_pvt["machine_code"].apply(add_bashotype)

df_pvt["bashoid"]=  df_pvt["machine_code"].apply(add_basho_id)


df_pvt["june"]= df_pvt[201706] / (0.1 + df_pvt[201703]  )    
df_pvt["may"]= df_pvt[201705] / (0.1 + df_pvt[201703]  )        
df_pvt.sort_values( "test_group", ascending=1, inplace=1)



df_pvt["nbuser"]=  df_pvt[201703].apply(lambda x: int(np.round(x/5.0, 0)) * 5)


df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/201706_pervm_usercount.csv', mode= "w" )
############################################################################################







########### per VM ##########################################################################
df_pvt= pd.pivot_table(df2, index= [ "machine_code"], columns=["yearmonth"],  
                       values= 'amount' ,   aggfunc= "sum" ).fillna(0).reset_index()  


df_pvt["test_group"]= df_pvt["machine_code"].apply(add_map_test)    
df_pvt["bashotype"]=  df_pvt["machine_code"].apply(add_bashotype)

df_pvt["bashoid"]=  df_pvt["machine_code"].apply(add_basho_id)


df_pvt["june"]= df_pvt[201706] / (0.1 + df_pvt[201703]  )    
df_pvt["may"]= df_pvt[201705] / (0.1 + df_pvt[201703]  )        
df_pvt.sort_values( "test_group", ascending=1, inplace=1)



df_pvt["nbuser"]=  df_pvt[201703].apply(lambda x: int(np.round(x/5.0, 0)) * 5)


df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/201706_pervm_sales2.csv', mode= "w" )
############################################################################################



vlist= [  6036,6041,7766,19404,21201,22974,27799,27802,29809,29810,29856,31724,32871,32984,34313,34462,35063,35322,40111,40124,40146,40172,40175,40182,40249,40679,40680,40681,40712,40872,40882,41042,41049,41128,41199,41213,41224,41295,41436,41441,41568,41569,41687,41839,42499,42501,42502,42587,42606,42608,42612,42908,42909,42954,43031,43073,43074,43075,43076,43106,43107,43108,43113,43115,43116,43145,43148,43149,43150,43159,43160,43166,43187,43188,43193,43194,43220,43221,43232,43356,43361,43364,43449,43484,43485,43596,43675,43681,43682,43686,43793,43848,43940,43942,44224,44311,44373,44374,44422,44424,44443,44453,44489,44587,44694,44755,44835,44836,44848,44883,44884,44917,44924,44925,44929,44931,44933,44934,44935,44936,44939,44940,44941,44942,44951,44954,44957,44958,44960,44962,45019,45022,45090,45097,45103,45104,45107,45120,45153,45157,45158,45211,45241,45291,45387,45446,45518,45575,45981,46007,601046,612668,612974,613324,617810,913804,914027,926758,927518,927699,929359,929360,990694,991137 ]

df2= df1[ df1.js_code.isin(vlist)]




         
         
         
         
         
         
         
         
         
         




############################################################################################
df_pvt= pd.pivot_table(df1, index= [ "test_group", "machine_code"], columns=["yearmonth"],  
                       values= 'amount' ,   aggfunc="sum" ).fillna(0)  


df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/201706_pervm_sales.csv', mode= "w" )












########### Product Category / Product Size  ###############################################
df_product= pd.read_csv( uri.map + '/map_cokeon_productname_master.txt')
df_product.columns

map_productsegment= util.pd_df_todict(df_product, colkey= "js_code" , colval= "product_segment" )   
def add_productsegment(js_code) :
   try :      return map_productsegment[js_code]
   except :   return ''
df1["productsegment"]=  df1["js_code"].apply(add_productsegment ).astype('category') 






############################################################################################
##### R0 List
df_r0_kev= pd.read_csv( uri.map + '/map_r0_vm_list.txt').fillna(0)
df_r0_kev.dtypes
df_r0_kev["machine_code"]= df_r0_kev["machine_code"].astype("int64")
'''
  [0, 'WATER', 'NST', 'SPORTS', 'cafeolait', 'COFFEE', 'SSD']

'''


vm_list1= df_r0_kev[df_r0_kev.productype== 'WATER'   ].machine_code.unique()



df_r0_sai= pd.read_csv( uri.map + '/map_r0_vm_list_saitou.txt').fillna(0)
df_r0_sai.dtypes
'''
array(['BLACK TEA', 'COFFEE', 'COFFEE-CAFEOLAIT', 'LACTIC', 'SSD', 'TEA',
       'WATER'], dtype=object)
'''



vm_list1=  df_r0_kev.machine_code.unique()


vm_list1= df_r0_kev[df_r0_kev.productype== 'cafeolait'   ].machine_code.unique()
vm_list2= df_r0_sai[df_r0_sai.productype== 'COFFEE-CAFEOLAIT'   ].machine_code.unique()


vm_list3= np.array( list(set(vm_list1).intersection( set(vm_list2) )) )



df2["js_code"]= df2["js_code"].astype('int64')



###### Monthly sales 

df2= df1[ (df1.js_code.isin([ 43449,43220,43221,40175,41687,44884,44951,44836,45211,45446,40172,40712,45241  ]) ) ]

                  
df2= df2[ (df2.machine_code.isin(vm_list1) ) ]
         
df2= df2[ -df2.machine_code.isin(vm_list1)  ]

         
       
         
df2= df1[ (df1.machine_code.isin(vm_list3) ) ]
         
df2= df2[  ( df2.productsegment== "COFFEE-CAFEOLAIT" )  ]
         


                    
df_pvt= pd.pivot_table(df2, index= [ "productsegment" ], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' ).fillna(0)                  
     


df_pvt= pd.pivot_table(df2, index= [  "machine_code"], columns=["yearmonth"],  
                          values= 'user_id' ,   aggfunc=pd.Series.nunique ).fillna(0)  


####Find user bought in march:
user_march=    df2[  df2.yearmonth== 201703].user_id.unique()
df2["ismarch"]= df2["user_id"].apply(  lambda x : 1 if x in user_march else 0    )   



df_pvt= pd.pivot_table(df2, index= [  "ismarch"], columns=["yearmonth"],  
                          values= 'user_id' ,   aggfunc=pd.Series.nunique ).fillna(0)  



############################################################################################
                    
df_pvt= pd.pivot_table(df2, index= [ "machine_code" ], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' ).fillna(0)    





df_pvt= pd.pivot_table(df2, index= [  "js_code"], columns=["yearmonth"],  
                          values= 'user_id' ,   aggfunc=pd.Series.nunique ).fillna(0)  

#########################################################################################


















#########################################################################################
####Find user bought in march:  #########################################################
df_r0_kev= pd.read_csv( uri.map + '/map_r0_vm_list.txt').fillna(0)
df_r0_kev.dtypes
df_r0_kev["machine_code"]= df_r0_kev["machine_code"].astype("int64")
vm_list1=  df_r0_kev.machine_code.fillna(-1).unique()


df2= df1[ df1.machine_code.isin( vm_list1 )  ]

df2b= df1[ -df1.machine_code.isin( vm_list1 )  ]
         
         
         

jjscode= 40172
jlist= [ 40172,
40712,
45241,
45211,
41687,
40175,
43221,
43220,
44951,
44884,
44836,
43449,
45446  ]



df_pvt0= None
for jjscode in jlist :

  df3= df2[df2.js_code== jjscode ]

  user_march=      df3[  (df3.yearmonth== 201705) & (df3.js_code== jjscode )].user_id.unique()
  df3["ismarch"]=  df3["user_id"].apply(  lambda x : 1 if x in user_march else 0    )   

  df_pvt= pd.pivot_table(df3, index= [ "js_code", "ismarch"], columns=["yearmonth"],  
                            values= 'user_id' ,   aggfunc=pd.Series.nunique ).fillna(0)  


  if df_pvt0 is None :   df_pvt0= df_pvt
  else :                 df_pvt0=  pd.concat(( df_pvt0 , df_pvt ))
   


  
  
  
  
  
  
  


'''
40172
40712
45241
45211
41687
40175
43221
43220
44951
44884
44836
43449
45446




40175
40712

41687
43220
43221

43449
44836

44884
44951


45211
45241
45446
'''



df3= df2[df2.ismarch==1]
df3= df2[df2.yearmonth == 201704 ]


df3.dtypes



df3.groupby("js_code").agg(  { "user_id":  pd.Series.nunique  })

         
df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/group_ref_e_all_transactions.csv', mode= "w" )





'''
1 User can buy in other VM, double count


'''





#############################################################################################         
df2_user_id= df2.user_id.unique()

df2_user_sales= df1[  df1.user_id.isin( df2_user_id ) ]


   

############################################################################################
###### All Transactions from Users, Month 4, Month 5   #####################################
df2= df1[ (df1.test_group == '') & ( df1.bashotype == "e.")  ]
df2_user_id= df2.user_id.unique()

df2_user_sales= df1[  df1.user_id.isin( df2_user_id ) ]

                    
df_pvt= pd.pivot_table(df2_user_sales, index= [  "user_id"], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' )                    
                    
df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/group_ref_e_all_transactions.csv', mode= "w" )




############################################################################################
######### Product bought   #################################################################
df2= df1[ (df1.test_group == '') & ( df1.bashotype == "e.")  ]
df2_user_id= df2.user_id.unique()

df2_user_sales= df1[  df1.user_id.isin( df2_user_id ) ]

                    
df_pvt= pd.pivot_table(df2_user_sales, index= [ "productsegment"], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' ).fillna(0, inplace=0)                    




`

df_pvt= df_pvt[ df_pvt[201705]> 0 ]
                    
df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/group_C_b_productsplit_all_transactions.csv', mode= "w" )




############################################################################################
######### Product bought   #################################################################
'''
df1= df1[ df1.yearmonth > 201702 ]

'''

df2= df1[ (df1.test_group == '')  & ( df1.bashotype == "e.")   ]
             
df_pvt= pd.pivot_table(df2, index= [ "productsegment",  "user_id"], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' )    
                
df_pvt= df_pvt[ ( (df_pvt[201705]> 0)  |  (df_pvt[201703]> 0)) ]
               
df_pvt.to_csv(r'G:/_data/raw/cokeon_csv/obppc_check/group_ref_e_user.csv', mode= "w" )


############################################################################################










############################################################################################
############################################################################################
vm_list1= df1[df1.yearmonth==201703].machine_code_.unique()



df2= df1[ (df1.test_group == '')  ]
df2= df2[ (df2.bashotype.isin(["a.", "b.", "e."]) ) ] 

####Find user bought in march:
user_march=    df2[  df2.yearmonth== 201703].user_id.unique()
df2["ismarch"]= df2["user_id"].apply(  lambda x : 1 if x in user_march else 0    )   


df_pvt= pd.pivot_table(df2, index=   ["test_group", "bashotype", "ismarch"], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=  pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)



df3= df2[df2.yearmonth == 201705 ]
df_pvt= pd.pivot_table(df3, index=   [ "test_group", "bashotype", "ismarch", "product_name"  ], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=  pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)





 "productsegment"
"js_code"






############################################################################################
dfc= df1[ (df1.test_group == 'C')  ]
dfc= dfc[ (dfc.bashotype.isin([ "e."]) ) ] 
user_march=    dfc[  dfc.yearmonth== 201703].user_id.unique()
dfc["ismarch"]= dfc["user_id"].apply(  lambda x : 1 if x in user_march else 0    )   

user_feb=     dfc[  dfc.yearmonth < 201703].user_id.unique()
dfc["isfeb"]= dfc["user_id"].apply(  lambda x : 1 if x in user_feb else 0    )   



user1= df1[ (df1.test_group != 'C')    ].user_id.unique()
dfc["isother"]= dfc["user_id"].apply(  lambda x : 1 if x in user1 else 0    ) 




df3= dfc

df3= dfc[dfc.yearmonth == 201703 ]
df_pvt= pd.pivot_table(df3, index=   [ "test_group", "bashotype", "productsegment"], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)



df_pvt= pd.pivot_table(df3, index=   [ "test_group", "bashotype", "ismarch", "product_name"], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)


          

df_pvt= pd.pivot_table(df3, index=   [ "test_group", "bashotype", "js_code"], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)

          


df_pvt= pd.pivot_table(df3, index=   [ "test_group", "bashotype", "ismarch", "isother" ], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)

          
          
          




############################################################################################
##### Product in March
df2= df1[ (df1.test_group == 'C')  ]
df2= df2[ (df2.bashotype.isin(["a.", "b.", "e."]) ) ] 

         
         
         
####Find user bought in march:
user_march=    df2[  df2.yearmonth== 201705].user_id.unique()
df2["ismarch"]= df2["user_id"].apply(  lambda x : 1 if x in user_march else 0    )   


df_pvt= pd.pivot_table(df2, index=   ["test_group", "bashotype", "ismarch"], 
                            columns= ["yearmonth"],  
                            values='user_id',      aggfunc=pd.Series.nunique)

df_pvt.fillna(0.0, inplace=True)









############################################################################################
############################################################################################
df_pvt=  df_pvt.loc[(df_pvt.sum(axis=1) != 0), (df_pvt.sum(axis=0) != 0)]
df_pvt.reset_index(inplace=1)





user_may=     df2[  df2.yearmonth== 201705].user_id.unique()
df2["ismay"]= df2["yearmonth"].apply(  lambda x : 1 if x in user_may else 0    )









############################################################################################
############################################################################################
df1= df1[ (df1.yearmonth > 201701 )   ]

         

         

############################################################################################         
############################################################################################         
gc.collect()         
         
         
       

  
         


         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         

"a."
"b."
"e."



df2.bashotype.unique()

[l., e., b., d., c., ..., f., h., 0, j., ]












############################################################################################
###### H0 test and pivot  by machine, montly transaction amount sales
df2= df1[ df1.test_group != '' ]

df_pvt= pd.pivot_table(df2, index= ["test_group", "bashotype", "machine_code"], columns=["yearmonth"],  
                          values='amount', aggfunc='count' )
df_pvt.fillna(0.0, inplace=True)
df_pvt=  df_pvt.loc[(df_pvt.sum(axis=1) != 0), (df_pvt.sum(axis=0) != 0)]
df_pvt.reset_index(inplace=1)











##### All area outside of  (test Y and Z)  
df3= df1[ df1.test_group == '' ]


df_pvt3= pd.pivot_table(df3, index= ["bashotype", "machine_code"], columns=["yearmonth"],  
                          values='amount', aggfunc='count' )
df_pvt3.fillna(0.0, inplace=True)
df_pvt3=  df_pvt3.loc[(df_pvt3.sum(axis=1) != 0), (df_pvt3.sum(axis=0) != 0)]
df_pvt3.reset_index(inplace=1)


df_pvt3.to_csv( r'G:/_data/raw/cokeon_csv/pivot_outsideVM.csv' )



##### Product by machine
df2["productype"]= df2["js_code"].apply( add_productype )

df3= df2[ (df2.yearmonth > 201703 )  & (df2.yearmonth < 201705)   ]



df_pvt5= pd.pivot_table(df3, index= ["test_group", "bashotype"], columns=["productype"],  
                           values='amount', aggfunc= sum)


df_pvt5.fillna(0, inplace=1)





df_pvt5= pd.pivot_table(df3, index= ["machine_code"], columns=["productype"],  
                           values='amount', aggfunc= sum)


df_product= pd.read_csv( uri.map + '/map_cokeon_productname_master.txt')

map_productypesize= util.pd_df_todict(df_product, colkey= "js_code" , colval= "product_segment" )  
def add_productype(js_code) :
   try :      return map_productypesize[js_code]
   except :   return ''

      
   
   
########   
df3["productype"]= df3["js_code"].apply( add_productype )



df3.head(10)



len( df3.machine_code.unique()  )














#############################################################################################
###### H0 test and pivot  and montly active  user
df2= df1[ df1.test_group != '' ]
df_pvt= pd.pivot_table(df2, index= ["test_group", "bashotype", "machine_code"], columns=["yearmonth"],  
                          values='user_id', aggfunc=pd.Series.nunique)
df_pvt.fillna(0.0, inplace=True)
df_pvt=  df_pvt.loc[(df_pvt.sum(axis=1) != 0), (df_pvt.sum(axis=0) != 0)]
df_pvt.reset_index(inplace=1)


##### All area outside of  (Y and Z)
df3= df1[ df1.test_group == '' ]


df_pvt3= pd.pivot_table(df3, index= ["bashotype", "machine_code" ], columns=["yearmonth"],  
                          values='user_id', aggfunc=pd.Series.nunique )
df_pvt3.fillna(0.0, inplace=True)
df_pvt3=  df_pvt3.loc[(df_pvt3.sum(axis=1) != 0), (df_pvt3.sum(axis=0) != 0)]
df_pvt3.reset_index(inplace=1)


df_pvt3.to_csv( r'G:/_data/raw/cokeon_csv/pivot_outsideVM_user.csv', mode="w" )




############################################################################################
###### H0 test and pivot  by machine, montly transaction amount sales
df2= df1[ df1.test_group != '' ]

df_pvt= pd.pivot_table(df2, index= ["test_group", "bashotype", "machine_code"], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' )
df_pvt.fillna(0.0, inplace=True)
df_pvt=  df_pvt.loc[(df_pvt.sum(axis=1) != 0), (df_pvt.sum(axis=0) != 0)]
df_pvt.reset_index(inplace=1)


##### All area outside of  (test Y and Z)  
df3= df1[ df1.test_group == '' ]


df_pvt3= pd.pivot_table(df3, index= ["bashotype", "machine_code"], columns=["yearmonth"],  
                          values='amount', aggfunc='sum' )
df_pvt3.fillna(0.0, inplace=True)
df_pvt3=  df_pvt3.loc[(df_pvt3.sum(axis=1) != 0), (df_pvt3.sum(axis=0) != 0)]
df_pvt3.reset_index(inplace=1)


df_pvt3["basho_id"]= df_pvt3["machine_code"].apply(add_basho_id)

df_pvt3.to_csv( r'G:/_data/raw/cokeon_csv/pivot_outsideVM_sum.csv' , mode="w")






#####################################

df1.to_csv( r'G:/_data/raw/cokeon_csv/cokeon_2016]201705.csv' , mode="w")











len( df1[ df1.basho_id > 0] )





gc.collect()




dfi0['purchased_at']

############################################################################################
max( df.dateday )


len(df[  df.dateday > 20170411 ])

df= df[  df.dateday < 20170411 ]


df['dateday']= df['purchased_at'].apply( lambda x: int(x[0:4] + x[5:7] + x[8:10] ) )




max( df.dateday )

arrow.get()

x= "2017/6/4 9:57"


df= df[ [ u'machine_code',
        u'js_code', u'product_name', u'price', u'temperature',
        u'amount', u'purchased_at', u'user_id',  u'day',
       u'month', u'year', 'dateday', 'basho_id']]

       
       
def day(s):    return int(s[8:10])
def month(s):  return int(s[5:7])
def year(s):   return int(s[0:4])
       


aa= dfi0.purchased_at.unique()
############################################################################################








df.to_hdf(uri.raw + r'/cokeon_csv/cokeon_2016_201705.hdf', 'data', mode='w', format="table")






gc.collect()




dfi0.columns



df.columns





###############    #########################################################################                
df= pd.read_csv(uri.rawcokeon+'/purchasings_BIG_CCWJ.csv', dtype= dtype0, nrows=10**8)       
gc.collect()  ;     util.pd_info(df)

df.columns




############################################################################################
df_fsmaster= pd.read_csv( uri.map + '/map_fsmaster_unerry.txt').fillna('0')
df_fsmaster.columns
''' Index([u'machine_code', u'cokeonarea', u'customer_name', u'basho_id',
       u'basho_name', u'inout', u'basho_type', u'Business condition code',
       u'basho_type_name', u'address', u'address_nb', u'Unnamed: 11',
       u'Unnamed: 12', u'Unnamed: 13', u'Unnamed: 14', u'Unnamed: 15'],
'''
df_fsmaster["machine_code"]=  df_fsmaster["machine_code"].astype('int64') 
df_fsmaster["basho_type2"]=   df_fsmaster["basho_type"].apply( lambda x : x[0:2] )
df_fsmaster["basho_id"]=  df_fsmaster["basho_id"].astype('int64') 


map_basho=     util.pd_df_todict(df_fsmaster, colkey= "machine_code" , colval= "basho_type2" )  

map_basho_id=     util.pd_df_todict(df_fsmaster, colkey= "machine_code" , colval= "basho_id" )  




def add_basho(x) :
   try :      return map_basho_id[x]
   except :   return -1
df["basho_id"]= df["machine_code"].apply(add_basho)


 
def add_basho2(x) :
   try :      return map_basho[x]
   except :   return -1
df["bashotype"]= df["machine_code"].apply(add_basho).astype('category') 



df[ (df.year > 2016 ) & (df.month > 3) & (df.day > 22) ]






#  df.to_csv(uri.rawcokeon+'/purchasings_BIG.csv')  




##### USE FSmaster to locate the positions   ###############################################
df= pd.read_csv(uri.rawcokeon+'/purchasings_FEB.csv', dtype= dtype0, nrows=10**8)  

df= df2




#####  Add Column   ########################################################################
map_ccwj= da.csv_todictmap(uri.map + 'map_ccwj_fsmaster.csv') 
map_ccwj2= map_ccwj['value']


def col_new(x):
  try :     return map_ccwj2[ float(x) ]
  except :  return 0


  
######  Location based on Machine Code   
def col_is_ccwj(x):
  try :    
     return map_ccwj2[ float( x) ]
  except :  
     return 0

df['is_ccwj']= df[ 'machine_code'   ].apply(col_is_ccwj)    




####Local Basho
map_basho_localcode=  da.csv_todictmap(uri.map + '/map_basho_tolocalbasho.txt')['localbashocode']
def addcol_localbasho(x) :
   try :      return  map_basho_localcode[x]
   except :   return -1

df['local_basho']=   df['basho_id'].apply(addcol_localbasho)
len( df['local_basho'].unique() )


######Local Basho
df2=   df[   df['local_basho']==    7312001  ]



df_pivot1= pd.pivot_table(df2, index="basho_id", columns='product_type', 
                          values='total', aggfunc='sum' )
df_pivot1.fillna(0.0, inplace=True)






#### df.columns





  
##### Filtering by bottler
def col_is_ccwj_bottler(x):
  try :    
     y=  x['bottler_code']
     if y== 49 or y== 44  or y== 33  or  y== 32   or  y== 34  : 
        return 1.0
     else :
        return 0.0
  except :  
     return 0.0

     
df['is_ccwj2']= df[[ 'machine_code', 'bottler_code' ]  ].apply(col_is_ccwj, axis=1)    




##############  FS Master   ################################################################
df_ccwj= df2[ df2.is_ccwj==1.0  ]
len(df_ccwj.machine_code.unique() )



df_missing=  df_ccwj[ df_ccwj.is_ccwj== 0.0  ]
len(df_missing.machine_code.unique() )


df_missing.bottler_code.unique()





df['is_ccwj']= df['machine_code'].apply(col_new)    
    
  
     
df['total']= df['amount'] * df['price']
print  df.columns  ;  df.head()

# df.to_csv(uri.rawcokeon+'/purchasings_BIG.csv', index=False)




 
####################### Merge of  Pandas Dataframe #########################################
df2.sort_values(['user_id','purchased_at'], inplace=True) 
 
df.sort_values(['user_id','purchased_at'], inplace=True) 

df= df.reset_index()
df2= df2.reset_index()



df2['purchased_at'].values[11455840], df['purchased_at'].values[11455840]


v2= df2['purchased_at'].values
v1= df['purchased_at'].values

v3= df2['user_id'].values
v4= df['user_id'].values


#Comparison  Check  
for ii in xrange(0, 11455845) :
   if v2[ii] != v1[ii]  or   v4[ii] != v3[ii]   :
       print ii,
       

df3= df[ [ 'user_rank_vol',  'day',  'month',  'year', 'hour',  'season',
       'daytime', 'weekday', 'product_type',  'product_brand']]

        
df5= pd.concat((df2, df3), axis=1)       

#  df.to_csv(uri.rawcokeon+'/purchasings_BIG.csv')       



####Hiearchical distance model
####  Cluster ---> Distance
  



dfi= df[ (df.machine_code== 9277078854) & (df.product_type=='SSD') ].amount.sum()


############################################################################################
##### total ################################################################################
dfi1= df[   df.user_id==   40618  ]
dfi2= df2[  df2.user_id==  40618  ]
dfi5= df5[  df2.user_id==  40618  ]


df5.columns       
df5.to_csv(uri.rawcokeon+'/purchasings_BIG.csv', index=False)
df= df5



df['total']= df['amount'] * df['price']





############################################################################################
################ Mapping: Product_name        ##############################################
map_product_cate= da.csv_todictmap(uri.map + 'map_cokeon_productname_master.txt') 
map_product_cate['brand_en_name']['ジョージア エスプレッソブレンド ギフト']
map_product_cate['product_type']['ｺｶ･ｺｰﾗ']
 


#### Add New Columns  ######################################################################
def col_brand(x):
  try :      return map_product_cate['brand_en_name'][x]
  except :   return ''

df['product_brand']= df['product_name'].apply( col_brand )
print  df.columns
df.head()



#####    Add items  ########################################################################
def col_product_type(x):
  try :      return  map_product_cate['product_type'][x]
  except :   return  ''

df['product_type']= df['product_name'].apply( col_product_type )
print  df.columns
df.head()


df.product_type.unique()
df_missing= df[ df.product_type == '' ]


df_missing1= df_missing[['product_name', 'js_code' ]].drop_duplicates()
df_missing1= list(df_missing1)


del df['Unnamed: 0']
df.columns


df.to_csv(uri.rawcokeon+'/purchasings_20170206_BIG.csv', index=False)
############################################################################################
############################################################################################



 

################ Mapping:  User_Group        ##############################################
map_user_cate= da.csv_todictmap(uri.raw + '/user_csv/user_BIG.csv') 


def col_user_type1(x):
  try :      return  map_user_cate['amt_week1'][x]
  except :   return  ''


df_mi['user_amt_week1']= df_mi['user_id'].apply( col_user_type1 )


print  df.columns
df.head()


#  df.to_csv(uri.rawcokeon+'/purchasings_BIG.csv', index=False, mode='w')
#  gc.collect()  


df_filter1.iloc[0, :  ]



df.product_type.unique()
# 'ENERGY', 'SPORTS', 'OTHER', 'SSD', 'WATER', 'LACTIC', 'COCA', 'COFFEE', 'NST', 'JUICE', 'BLACK TEA']




#####  Basho Code  ########################################################################
map_bashoid= da.csv_todictmap(uri.map + 'map_fsmaster_unerry_small3.txt')['areacode']

def col_bashoid(x):
  try :     return int(map_bashoid[float(x)])
  except :  return -1

 
df['basho_id']= df['machine_code'].apply( col_bashoid )

#  df.to_csv(uri.rawcokeon+'/purchasings_20170206_BIG.csv', index=False)  





  
####  Cafe au Lait  ########################################################################
'ddd5'.find('55')

map_bashoid['areacode']








############################################################################################
####### Ranking User : New Product placement     ###########################################
df= pd.read_csv(uri.rawcokeon+'/purchasings_20170206_BIG.csv', dtype= dtype0, nrows=10**8)       





### choose a product cluster ###############################################################
product0= 'COFFEE'
   
df_filter1= df[  (df.product_type == product0)  &  (df.is_ccwj== 1)  ] 

               
               
product0= 'cafe_NOLait'
def addcol_product(x):
 if  x.find(product0) > -1 :   
    return 1
 else :
    return 0
    
df['iscafelait']=  df['product_name'].apply( lambda x : addcol_product(x) )
    

df_filter1= df[  (df.iscafelait != 1 )  &   (df.product_type== 'COFFEE')  & (df.is_ccwj== 1)       ] 

         
df_filter1= df[  (df.is_ccwj== 1)       ] 

         
               
               
############################################################################################               
#### Top user of this cluster ##############################################################           
df_filter1_user= df_filter1.groupby('user_id').apply(peruser_stat_all)         # 100k users: 100s,
df_filter1_user.sort_values(by=['amount_sum'], ascending=0, inplace=True)


df_filter1_user.to_csv(uri.raw + '/newproduct_csv/user_'+product0+'_CCWJ_0206.csv')


df_filter1_user1= df_filter1_user[df_filter1_user.amount_sum > 10 ]

user_list1=  np.array(df_filter1_user.index[0:])  # np.array(df_filter1_user.index[0:5000])

df_topuser= df_filter1[  df_filter1['user_id'].isin( user_list1)]

                       
###  Machines for top users  2845  machines  ###############################################
mm_topuser= df_topuser.machine_code.unique() 




############################################################################################
### 'Big Fan' VM machines  #################################################################
df_filter1_top= df_filter1[  df_filter1['machine_code'].isin( mm_topuser   )]
                       
                       
df_filter1_machine= df_filter1_top.groupby('machine_code').apply(permachine_stat_all)    # 100k users: 100s,
df_filter1_machine['ratio_peruser']= df_filter1_machine['amount_sum'] / df_filter1_machine['nb_user']

df_filter1_machine= df_filter1_machine[ df_filter1_machine.amount_sum  > 5]


df_filter1_machine.sort_values(by=['ratio_peruser'], ascending=0, inplace=True)


df_filter1_machine['machine_code']= df_filter1_machine.index

df_filter1_machine['basho_id']= df_filter1_machine['machine_code'].apply(col_bashoid )



df_filter1_machine.to_csv(uri.raw + '/newproduct_csv/machine_'+product0+'_CCWJ_top_0206.csv')


############################################################################################
############################################################################################





############################################################################################
###  VM machines  #################################################################
                              
df_filter1_machine= df_filter1.groupby('machine_code').apply(permachine_stat_all)    # 100k users: 100s,
df_filter1_machine['ratio_peruser']= df_filter1_machine['amount_sum'] / df_filter1_machine['nb_user']
df_filter1_machine= df_filter1_machine[ df_filter1_machine.amount_sum  > 5]

df_filter1_machine['machine_code']= df_filter1_machine.index
df_filter1_machine['basho_id']= df_filter1_machine['machine_code'].apply(col_bashoid )


df_filter1_machine.to_csv(uri.raw + '/newproduct_csv/machine_CCWJ_top_0206.csv')










#### 20118    ##############################################################################
len(df_filter1.machine_code.unique() )


#### Total CCWJ Machines:   56073 VM   vs  120546 VM    ####################################
##  len( df[ (df.is_ccwj== 1)  ].machine_code.unique() )   
##  len( df.machine_code.unique() )   




############################################################################################
############################################################################################
############################################################################################



#############Test     
#### Machine Profile  ######################################################################
df_mi=  df[  df.machine_code.isin( [ 
9300022912,
9266101076,
9300074353,
9266093220,
9300089744,
9300064111,
9300035077,
9300035753

]   ) ]

                                    
                              
df_mi_productype= df_mi.groupby(['machine_code', 'product_type']).apply(permachine_stat_all)  






############################################################################################
df_mi_usertype= df_mi.groupby('user_amt_week1').apply(permachine_stat_all)  





def permachine_stat_all(dfi) :
 dfi.sort_values('purchased_at', inplace=True)
 nb_user= len(dfi.user_id.unique())
 nb_drink=   len(dfi.product_name.unique())
 nb_transaction=          len(dfi)
 nb_product_type=   len( dfi.product_type.unique() )
 
 datei= dfi.purchased_at.values  
 t0, t1=  datei[0], datei[-1]
 dt=      date_diffsecond(t1 , t0 ) 
 amount_sum=  dfi.amount.sum() * 1.0
 total= dfi.total.sum() * 1.0  
 
 d= [amount_sum, nb_user, nb_drink, nb_transaction, nb_product_type, bottler_code,
     t0, t1, dt, total 
     ]
 i= ['amount_sum', 'nb_user', 'nb_product', 'nb_transaction', 'nb_product_type',
     'bottler_code', 't0', 't1', 'period_sec',  'total' ]
 return pd.Series(data=d, index=i)
  

 
########## Evolution of cokeon Machines   ##################################################
def permonth_stat_all(dfi) :
  # dfi.sort_values('purchased_at', inplace=True)
  nb_user=        len(dfi.user_id.unique())
  nb_machine=     len(dfi.machine_code.unique())
  amount_sum=  dfi.amount.sum() * 1.0
  total= dfi.total.sum() * 1.0  

  d= [nb_user, nb_machine, amount_sum, total  ]
  i= [ 'nb_user', 'nb_machine', 'amount_sum', 'total'  ]
  return pd.Series(data=d, index=i)
  
 
  
df_ccwj=  df[  (df.bottler_code== 34 )  ] 
   
df_month_ccwj= df_ccwj.groupby(['month']).apply( permonth_stat_all )  



df_month= df.groupby(['month']).apply( permonth_stat_all )  




###### Data :               ################################################################
df_ccwj=  df[  (df.bottler_code== 49 )  ] 
             
             
             
df_ccwj= df[ df.is_ccwj2==1.0  ]




len(df_ccwj.machine_code.unique())
# 49903
             
####
len(df.machine_code.unique())

      

df_ccwj=  df_ccwj[  df_ccwj.month < 10 ] 



##### 
df_ccwj=  df[  (df.bottler_code== 49 ) & (df.month < 6)   ] 



df_ccwj=  df[  (df.month == 12 )   ] 




########        ############################################################################        
df_ccwj= df             
             
df_month_ccwj= df_ccwj.groupby(['bottler_code']).apply( permonth_stat_all )  


df_month_ccwj.head()



df_month_ccwj= df.groupby(['month']).apply( permonth_stat_all )  


df_loc= df[['area_code', 'bottler_code', 'bottler_name' ]]
df_loc.drop_duplicates(   inplace= True    )



df1= df[  (df.bottler_code == 32 )  ]
             
             




        
        
        
############################################################################################
df_mi2=  df[  df.machine_code.isin( [  9266139319  ]   ) ]

            

df_mi2=  df[  df.machine_code.isin( [   9266072294 ]   ) ]


            
a= df.user_id.unique()

loc= pd.DataFrame( np.array(df.bottler_name.unique()) , columns=['id'])


loc_code= pd.DataFrame( np.array(df.bottler_code.unique()) , columns=['id']).values



loc_code= df.bottler_code.unique()


for x in loc_code :
   print x, df[ df.bottler_code == x ]['bottler_name'].values[0]

   

   
   
   
   
   
   


############################################################################################
######## Summer Selection   ################################################################
df_mm_list=   df[  ( df.month < 7  ) &  (df.bottler_code== 49)  ]

                 
df_summer1=   df[  ( df.month < 10  ) &  (df.bottler_code== 49)  ]



df_machine= df_summer1.groupby('machine_code').apply(permachine_stat_all)     #  100k users: 100s,
df_machine.sort_values([  'amount_sum' ], ascending=False, inplace=True)


df_machine.to_csv(uri.raw+'/machine_summer_selection_01.csv', mode='w')

################     #######################################################################

  
















############################################################################################
###Brand List :  array(['I LOHAS', 'Toreta!'],
df_filter1.product_brand.unique()

prod_list= df_filter1.product_name.unique()
for x in prod_list: print x
   






### Filtering with Top machines           
           


           
         
           
           

############################################################################################
### Per User stats #########################################################################
def peruser_stat(dfi) :
 nb_brand=   len(dfi.product_brand.unique())
   
 d= [nb_brand ]
 i= ['nb_brand' ]
 return pd.Series(data=d, index=i)
  
 
df_user= df.groupby('user_id').apply(peruser_stat)         # 100k users: 100s,
df_user.to_csv(uri.raw+'/user_csv/user_ALL_stat_002.csv')



##### LTV per User #########################################################################
dfi= df.loc[ df.user_id==  40618  ]

t1= str( datetime.now() )
t1= '2016-12-18 00:00:00'

dfi['diff_fromtoday_sec']=   dfi['purchased_at'].apply( lambda x: date_diffsecond(str_t1=t1, str_t0=x)  )



ltv_90d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 90 ].total.sum()
ltv_60d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 60 ].total.sum()
ltv_30d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 30 ].total.sum()




def peruser_stat(dfi) :
 dfi['diff_fromtoday_sec']=   dfi['purchased_at'].apply( lambda x: date_diffsecond(str_t1=t1, str_t0=x)  )

 ltv_90d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 90 ].total.sum()
 ltv_60d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 60 ].total.sum()
 ltv_30d= dfi[ dfi['diff_fromtoday_sec'] < 86400 * 30 ].total.sum()
   
 d= [ ltv_30d, ltv_60d, ltv_90d    ]
 i= [ 'ltv_30d', 'ltv_60d', 'ltv_90d'   ]
 return pd.Series(data=d, index=i)
  

 
df_user= df.groupby('user_id').apply(peruser_stat)         # 100k users: 100s,
df_user.to_csv(uri.raw+'/user_csv/user_ALL_stat_LTV.csv')

############################################################################################







############################################################################################
df_product= df[['product_name',  'product_type']].groupby('product_name')         # 100k users: 1

df_product= df_product.apply(lambda x : 0)



def perproduct_stat(dfi) :  
  d= [dfi.product_name.values[0], dfi.product_type.values[0]   ]
     
  i= ['product_name',  'product_type'  ]
    
  return pd.Series(data=d, index=i)
 
 
df_product= df.groupby('product_name').apply(perproduct_stat)

df_product.to_csv(uri.map + '/map_productname_category2.txt')


  


print '\xe3\x82\xb8\xe3\x83\xa7\xe3\x83\xbc\xe3\x82\xb8\xe3\x82\xa2 \xe3\x82\xa8\xe3\x82\xb9\xe3\x83\x97\xe3\x83\xac\xe3\x83\x83\xe3\x82\xbd\xe3\x83\x96\xe3\x83\xac\xe3\x83\xb3\xe3\x83\x89 \xe3\x82\xae\xe3\x83\x95\xe3\x83\x88'








### Per User stats #########################################################################

 
df_user= df.groupby('user_id').apply(peruser_stat_all)            #  100k users: 100s,
df_user.to_csv(uri.raw+'/user_csv/user_ALL_stat.csv')             #  Details








###### CCWJ   
df= df[ df.bottler_code == 49 ]
#  df.to_csv(uri.rawcokeon+ 'purchasings_BIG_CCWJ.csv')




#### All in CCWJ   #########################################################################
user_all= df.user_id.unique() 



############################################################################################
#### Transaction  ----> USER / MACHINES table###############################################
df2.columns

df3= df2[[ 'js_code', 'product_name'   ]]


df4= df3.drop_duplicates()





####


#######Following tables  ###################################################################
USER_GROUP_0001
   is_more_1week
   is_more_1month
   is_more_3month
   is_more_2products
   freq_group1  : more 1week, low, medium, high
   location_group1  :
   machine_group1 :
      
   coffee_level:
   
   
USER_STAT_0001    # Total Stats, can be aggregated with new data
   list_product_name
   freq_product_name
   list_product_type
   freq_product_type
   list_machine_code
   freq_machine_code

   amount  : qty total
   total  : JPY
   t0
   t1
   period_second   
   avg_per_week
   amt_perweek
   total_perweek
   
   nb_coffee_perweek
   nb_soda_perweek
   nb_water_perweek
   
  nb_coffee_perweek
   nb_soda_perweek
   nb_water_perweek
    
   


USER_STAT_00A0    # Monthly stats:  12_months
   201604_total
   201604_amount
   201604_list_product_name
   201604_freq_product_name
   201604_list_product_type
   201604_freq_product_type
   201604_list_machine_code
   201604_freq_machine_code


USER_STAT_0A00    # Weekly stats:  3_months
   20160401_total
   20160401_amount
   20160401_list_product_name
   20160401_freq_product_name
   20160401_list_product_type
   20160401_freq_product_type
   20160401_list_machine_code
   20160401_freq_machine_code



   

   
   
   

   
   

    
   
   

### Per User stats #########################################################################
def peruser_stat_all(dfi) :
 dfi.sort_values('purchased_at', inplace=True)
 nb_machine= len(dfi.machine_code.unique())
 nb_drink=   len(dfi.product_name.unique())
 nb_total=          len(dfi)
 nb_product_type=   len(dfi.product_type.unique() )
 nb_location= len(dfi.location_code.unique())
 bottler_code= dfi.bottler_code.values[0]
 
 datei= dfi.purchased_at.values  
 t0, t1=  datei[0], datei[-1]
 dt=      date_diffsecond(t1 , t0 ) 
 amount_sum=  dfi.amount.sum() * 1.0
 total= dfi.total.sum() * 1.0  

 d= [amount_sum, nb_machine, nb_drink, nb_total, nb_product_type, 
     nb_location, bottler_code, t0, t1, dt, total ]
 i= ['amount_sum',  'nb_machines',  'nb_product_name', 'nb_transaction', 'nb_product_type', 
     'nb_location', 'bottler_code', 't0', 't1', 'period_sec', 'total']
 return pd.Series(data=d, index=i)
  
 
df_user= df.groupby('user_id').apply(peruser_stat)         # 100k users: 100s,
df_user.to_csv(uri.raw+'/user_csv/user_ALL_stat.csv')





### Per Machine stats ######################################################################
def permachine_stat_all(dfi) :
 dfi.sort_values('purchased_at', inplace=True)
 nb_user= len(dfi.user_id.unique())
 nb_drink=   len(dfi.product_name.unique())
 nb_transaction=          len(dfi)
 nb_product_type=   len(dfi.product_type.unique() )
 bottler_code= dfi.bottler_code.values[0]
 
 datei= dfi.purchased_at.values  
 t0, t1=  datei[0], datei[-1]
 dt=      date_diffsecond(t1 , t0 ) 
 amount_sum=  dfi.amount.sum() * 1.0
 total= dfi.total.sum() * 1.0  
 
 d= [amount_sum, nb_user, nb_drink, nb_transaction, nb_product_type, bottler_code,
     t0, t1, dt, total 
     ]
 i= ['amount_sum', 'nb_user', 'nb_product', 'nb_transaction', 'nb_product_type',
     'bottler_code', 't0', 't1', 'period_sec',  'total' ]
 return pd.Series(data=d, index=i)
  
 

df_machine= df.groupby('machine_code').apply(permachine_stat)     #  100k users: 100s,
df_machine.to_csv(uri.raw+'/machine_ALL_stat.csv', mode='w')


df_mach_ccwj=  df_machine[df_machine.bottler_code == 49]






############################################################################################
df_mach_ccwj.nb_user.mean()
10.42069289589474


da.col_study_distribution_show(df_mach_ccwj, col_include=['nb_user'])





#### Distribution of users among machines  #################################################
user_select=   user_all
#df_select=     df
user_label =  'user_all_ccwj_'









dftest= df[df.user_id < 5]


######User stats   #########################################################################
def peruser_stat_purchased(dfi) :  
  datei= dfi.purchased_at.values  
  t0, t1=  datei[0], datei[-1]
  dt=      date_diffsecond(t1 , t0 ) 
  amount_sum=  dfi.amount.sum() * 1.0

  d= [t0, t1, dt, amount_sum ]
  i= ['t0', 't1', 'date_sum', 'amount_sum']
  return pd.Series(data=d, index=i)


df_user= df_select.groupby('user_id').apply(peruser_stat_purchased)
# 56.1 s per loop, 57869 users: 50s,  100k users: 100s,
df_user.to_csv(uri.raw+'/user_csv/user_purchased.csv')




def pd_merge_column(df1, df2) :
  return pd.merge(df1, df2, left_index=True, right_index=True, how='outer')


def pd_update_values(df1, df2) :
  return df1.combine_first(df2)

  
  

  users[kk]['location_nb']=  len(list(dfi.location_code.unique()))
  users[kk]['location']=     np_list_tofreqdict(dfi.location_code.values)
  

  users[kk]['nb_last_week1']=  int(dfi[dfi.date_sec_end <  604800   ].amount.sum())
  users[kk]['nb_last_week2']=   int(dfi[dfi.date_sec_end <  604800*2 ].amount.sum())
  users[kk]['nb_last_week3']=   int(dfi[dfi.date_sec_end <  604800*3 ].amount.sum())
  users[kk]['nb_last_week4']=  int(dfi[dfi.date_sec_end <  604800*4 ].amount.sum())
  users[kk]['nb_last_week5']=  int(dfi[dfi.date_sec_end <  604800*5 ].amount.sum())

  
  

users= util.np_dictordered_create()
for kk in user_select :  users[kk]= {}


for ii, kk in enumerate(user_select) :
 if ii > -1 : 
  dfi= df_select[df_select.user_id == kk ]

  a=  dfi.product_name
  users[kk]['nb_machine']= len(dfi.machine_code.unique())
  users[kk]['nb_drink']=   len(a.unique())
  users[kk]['nb_total']=          len(a)
  users[kk]['nb_product_type']=   len(dfi.product_type.unique() )
  
  if ii %  10**3 == 0 :      print(str(ii))
  # if ii %  (5*10**4) == 0 :  util.save(users, user_label + '_' + str(ii) )



  
#-- Time Series purchase  ------------------------------------------------------------------
for ii, kk in enumerate(user_select) :
 if ii > -1 :   
  
   datei= dfi.purchased_at.values  
  t0, t1=  datei[0], datei[-1]
  
  users[kk]['t0']= t0
  users[kk]['t1']= t1

  dt=      date_diffsecond(t1 , t0 ) 
  users[kk]['date_total']= dt

  x=  dfi.amount.sum() * 1.0
  users[kk]['amount_total']=   x                        #  604800s in 1 week
#  util.save_test( user_label + '_' + str(ii) ) 




#Export dict to csv table ############################################################################################
csvexport= uri.out1 +  '/user/user_info2.csv'
ii=0
with open(csvexport, 'w') as f:
  for key, value in users.items() :
    ii+=1
    #if ii > 10 : break
    if ii==1 : f.write(  'user_id,' +  util.np_dict_tostr_key(value)  + '\n')
    f.write( '{0},{1}\n'.format(key, util.np_dict_tostr_val(value) ) ) 
print( csvexport )  
    
    


#### Merge 2 csv file by user_id








#### df_user
df_user= pd.read_csv(uri.rawuser + 'user_BIG.csv' )









    

############################################################################################
#### Transaction  ---->  BIG user_list   ###################################################
user_select=   user_all
df_select=     df_all
user_label =   'user_all_'


users= util.np_dictordered_create()
for kk in user_select :  users[kk]= OrderedDict()   # {}

for ii, kk in enumerate(user_select) :
  if ii < -1  : break 
   
  dfi= copy.deepcopy(df_select[df_select.user_id == kk ])
  # dfi.sort_values('purchased_at', inplace=True)
  del dfi['user_id']


  # User group :
  users[kk]['user_rank_vol']=  dfi['user_rank_vol'].values[0]

  #Time series
  datei= dfi.purchased_at.values  
  t0, t1=  datei[0], datei[-1]
  dt=      date_diffsecond(t1 , t0 ) 
  users[kk]['date_total']= dt
  users[kk]['t0']= t0
  users[kk]['t1']= t1

  users[kk]['date']= datei  
  users[kk]['date_sec_end']= np.array([ date_diffsecond(str_t1=t1, str_t0=x) for x in datei  ])


  #Put in dataframe
  dfi['date_sec_start']= dfi['purchased_at'].apply( lambda x: date_diffsecond(x, str_t0=t0)   )
  dfi['date_sec_end']=   dfi['purchased_at'].apply( lambda x: date_diffsecond(str_t1=t1, str_t0=x)  )

  
  x=  dfi.amount.sum() * 1.0
  users[kk]['amount_sum']= x
  # users[kk]['amount_perweek']=   x / ( dt / 6048005.0)  #  604800s in 1 week
  # users[kk]['amount_perday']=    x / ( dt / 86400.0)  #  604800s in 1 day
  # users[kk]['amount_permonth']=  x / ( dt / 18295200.0)  #  604800s in 1 month
  
  
  #Basket profile
  users[kk]['drink_nb']=  len(list(dfi.product_name.unique()))
  users[kk]['drink']=     np_list_tofreqdict(dfi.product_name.values)
  
  users[kk]['location_nb']=  len(list(dfi.location_code.unique()))
  users[kk]['location']=     np_list_tofreqdict(dfi.location_code.values)
  

  users[kk]['nb_last_week1']=  int(dfi[dfi.date_sec_end <  604800   ].amount.sum())
  users[kk]['nb_last_week2']=   int(dfi[dfi.date_sec_end <  604800*2 ].amount.sum())
  users[kk]['nb_last_week3']=   int(dfi[dfi.date_sec_end <  604800*3 ].amount.sum())
  users[kk]['nb_last_week4']=  int(dfi[dfi.date_sec_end <  604800*4 ].amount.sum())
  users[kk]['nb_last_week5']=  int(dfi[dfi.date_sec_end <  604800*5 ].amount.sum())
  
  # users[kk]['dfi']=  dfi

  if ii %  10**3 == 0 :      print(str(ii))
  if ii %  (5*10**4) == 0 :  util.save(users, user_label + '_' + str(ii) )
util.save(users, user_label + '_' + str(ii) )

  
util.save_test(user_label + '_' + str(ii) )
gc.collect()





for k,x in users.items() :
  v= users[k]
  print v['amount_sum'], v['amount_perweek'], v['nb_last_week1'], v['nb_last_week4'], v['nb_last_week3']
############################################################################################













############################################################################################
v= df_best['user_id'].values

  
  
  #  dfi.purchased_at.values.sort()   #sort the dates
  # Keep the dataframe
  users[kk]['purchased_at']= dfi.purchased_at.to_string(index=False)   #.replace('\n',',') 
  users[kk]['amount']=       dfi.amount.to_string(index=False)
  users[kk]['product_name']= dfi.product_name.to_string(index=False).replace('          ','')
  users[kk]['machine_code']= dfi.machine_code.to_string(index=False) 


  datei= dfi.purchased_at.values  
  t0=  datei[0]   
  t1=  datei[-1]
  users[kk]['t0']= t0
  users[kk]['t1']= t1   
  users[kk]['date_total']= date_diffsecond(t1 , t0 )    
  users[kk]['date_sec_start']= [ date_diffsecond(t, t0) for t in datei  ] # from Start
  users[kk]['date_sec_end']= [ date_diffsecond(t1, t) for t in datei  ]  # From end




users[kk]['date_sec_end'][-20]



date_diffsecond('2016-12-15 22:36:00', '2016-12-15 08:00:00') / (3600.)
   

  dfi.purchased_at.to_string
  
  
    
#### issues    
eval( str(dfi.product_name.values))
 print users[kk]['product_name'].split('\n')

  
n= len(df_best)
for i in xrange(0,n) :
   kk=  df_best.iloc[i, 7]
   #idate= arrow.get(df_best.iloc[i, 6], fmt )
   idate= df_best.iloc[i, 6]
   user_id[kk].append(idate)



###date, day creation    ##########################################################
for k, datek in user_id.items() :
    datek.sort()
    ndate= len(datek)
    dateday= np.zeros(ndate, dtype=np.int32)

    
    datek0=  arrow.get(datek[0], fmt)
    for i in xrange(0, len(datek)) :
       dateday[i]=   (arrow.get(datek[i], fmt) - datek0).days

    user_id[k]['date']= datek
    user_id[k]['day']=  dateday



##################################################################################
#### Data model to user





import copy
user_id2= copy.deepcopy(user_id)


def weekday(s, fmt='YYYY-MM-DD', i0=0, i1=10):
  s2= s[i0:i1]
  try :
    return  cache_weekday[s2]
  except KeyError:
    wd= arrow.get(s2, fmt).weekday()


    


#==============================================================================
####### Pivot Table on the CSV file   ###############################################
df_pivot1= pd.pivot_table(df,
  index='user_id' ,   columns='bottler_code', 
  values='amount', aggfunc='sum' )
df_pivot1.to_csv(out1+ 'byuser_bottler_amount.csv', index=True)





####### Pivot Table on the CSV file   ###############################################
df_pivot1= pd.pivot_table(df,
  index='user_id' ,   columns='product_name', 
  values='amount', aggfunc='sum' )


df_pivot1.fillna(0, inplace=True)
df_pivot1


file1= 'df_pivot1_user_product_amount'
df_pivot1.to_csv(out1+ file1+'.csv', index=True)
util.save(df_pivot1, out1+file1 + '.pkl')
gc.collect()



df_pivot1['product_name'] = df_pivot1.index

df_pivot1.columns


#### Pandas Pivo


df_pivot1.to_csv(out1+ file1+'.csv', index=False)
















#################################################################################################
###### Read file and extract data pattern:  .csv,  .txt, .xls  ##################################
file1= u'【OBPPCデータ】ロケチャネル×VPM.xlsx'
dir1= u'E:/_data/unerry/csv/vm_csv/20161208 OBPPC-1/'


file1= u'*.xlsx'
dir1= u'E:/_data/unerry/csv/vm_csv/'
da.xl_getschema(dir1, filepattern=file1, outfile=dir1+'/schema.xlsx')








#----- Extract zip to folder                            -----------------------------
#   util.os_zipextractall(data+'/zip/*.zip',  data+'/csv/')


#### unicode  --- > utf-8, before SAVING
df2['VALIDEND']= df2['VALIDEND'].apply(util.str_to_utf8)
df2['COUPON_ID_hash']= df2['COUPON_ID_hash'].apply(util.str_to_utf8)


######################## CSV Aanalysis    ###########################################
# Parse CSV Files: put Summary into Excel + Type_guess  --------------------------------
rawcokeon= 'E:/_data/unerry/csv/cokeon_csv/'
df_schema, df_type_guess_all=  da.csv_col_schema_toexcel(dircsv=rawcokeon, filepattern='*.csv',
                                  outfile=rawcokeon + '/_meta.xlsx',
                                  maxrow= 500000, returntable=1)

## 11 mio lines


df_schema, df_type_guess_all= util.load(rawcokeon+'_meta_schema.pkl'),  util.load(rawcokeon+'_meta_type_guess.pkl')



###### Update meta database  ########################################################
#  ALLDB= util.load( data+'ALL_DB_META.pkl')
#   ALLDB={'cokeon':  { } }

'''
ALLDB= da.db_meta_add(ALLDB, 'cokeon', schema=df_schema, df_table_uri=None, df_table_columns=None)
ALLDB['cokeon']['table_uri']
util.save(ALLDB, data+'ALL_DB_META.pkl')
  
ALL_schema, ALL_type_guess_all= util.load(rawcokeon+'_meta_schema.pkl'),  util.load(rawcokeon+'_meta_type_guess.pkl')

ALLDB= da.db_meta_add(metadb= ALLDB, dbname='cokeon', new_table= ('type_guess', ALL_type_guess_all)   )

ALLDB= da.db_meta_add(ALLDB, 'cokeon', df_table_uri=df_uri  )

ALLDB= da.db_meta_add(ALLDB, 'cokeon', df_table_columns=df_col)


'''

###Access to DB, table fromm ALLDB,  Look_up in all table  ##########################
#### Query the Meta Table  and get list of URI 
db_find_rs= da.db_meta_find(ALLDB, query='bottler')


db_find_rs=  da.db_meta_find(ALLDB, query='bottler.csv', filter_db=['cokeon'],   filter_table=['table'], filter_column=['table'] )

db_find_rs[0][0], db_find_rs[0][1], db_find_rs[0][2]


da.db_meta_find(ALLDB, query='bottler', filter_db=['cokeon'],   filter_table=['table'], filter_column=['table'] )


da.db_meta_find(ALLDB, query='bottler', filter_db=['cokeon'],   filter_table=['table'] )


da.db_meta_find(ALLDB, query='bottler', filter_db=['cokeon'])


#### Query Dataframe
ALLDB['cokeon']['table_uri'].query("table== '"+ table1 +"'")




#### Find Pattern in Pandas dataframe  ##############################################
util.pd_find(df, '08:31', doreturnposition=True )
util.pd_find(df, '08:29', col_restrict=['purchased_at'], doreturnposition=True )
util.pd_find(df, '08:29', col_restrict=['bottler_name'] )
util.pd_find(df, 100.0, isnumeric=True )


## 10mio Transact, 1.5Go,  find takes 5s to list all patterns
df_find, df_find_pos= util.pd_find(df, '08:31', doreturnposition=True )





#####################################################################################



################### CSV Files Transfomation   #######################################      
df= pd.read_csv(rawcokeon+ 'purchasings.csv', nrows=10**3)      
gc.collect()                    
df.head(),  util.pd_info(df)           
           
           
''' Transformation of data :

purchase_at --->  month, day, year, hour, weekday, daytime


location_code -->
   location__type, location__building_type1, location__building_type2, 
   location__inout, location__conbini_dist, location__info1, location__info2


product_name -->  
   product__brand_code,   product__brand_en_name,  categories__code, categories__abbr


Create  a view:
df_v1= df[[col1, col2, col3  ]] 
df_v1[newcol]=   
  


'''           

#2) Save the table in csv or in DB


#3) Process to Pivot table:  VIEWS


#4)  Generate Report


'''
Cluster of   (Machine, Users)
        of   (Users, Drink)

Marketing Campaign ---> Select Target Population
   system --> select machines  Targets
   Validation of user
   
   ---> Can generate report on Sales

Think of colums to be seen

'''

##### 0) Create Flat View in SQL join of all multi table columns ---> Generate the CSV
import datanalysis as da


#### 1) Test CSV and Find Dtypes    #################################################
table1= 'purchasings'




##### ressource locator   ###########################################################
url1= rawcokeon+ 'purchasings.csv'    #Be careful of DASK
url1= ALLDB['cokeon']['table_uri'][table1]


df= pd.read_csv(ALLDB['cokeon']['table_uri'][table1], nrows=10**3)       
gc.collect()                   
df.head(),  util.pd_info(df)     

util.pd_dtypes(df, returnasdict=0)


ALLDB['cokeon']['type_guess']['purchasings']

dtype0= {'amount': 'int16',
 'area_code': 'int32',
 'bottler_code': 'int16',
 'bottler_name': 'category',
 'id': 'int32',
 'js_code': 'int32',
 'location_code': 'int32',
 'machine_code': 'int64',
 'price': 'int16',
 'product_name': 'category',
 'promotion_id': 'float16',
 'purchased_at': 'object',
 'temperature': 'int16',
 'user_id': 'int32'}


df= df.astype(dtype0)





#### 1) Read CSV FULL: partial columns,   ###########################################
usecols=["user_id", "machine_code", "location_code",  "area_code", 'product_name', 
         'temperature', "amount",'purchased_at']

df= pd.read_csv(rawcokeon+ 'purchasings.csv', dtype= dtype0, usecols= usecols, nrows=10**8)       
gc.collect()       
util.pd_info(df)     







#### Create view with new columns
df_v1= df.loc[:,("user_id", "machine_code", "location_code",  "area_code", 'product_name', 
              'temperature', "amount",'purchased_at')]
# df_v1= df[['machine_code', 'location_code', 'purchased_at' ]]   #Error
df_v1.dtypes
del df; gc.collect()

df_v1.iloc[0,2]

 

### Add Date  ----------------------------------------------------------------------
import arrow
###################Faster one   #####################################################
#'YYYY-MM-DD    HH:mm:ss'
#"0123456789_10_11
def day(s):    return int(s[8:10])
def month(s):  return int(s[5:7])
def year(s):   return int(s[0:4])
def hour(s):   return int(s[11:13])
# def weekday(s):  return arrow.get(s, 'YYYY-MM-DD HH:mm:ss').weekday()



###Super Fast because of caching
cache_weekday= {}
def weekday(s, fmt='YYYY-MM-DD', i0=0, i1=10):
  s2= s[i0:i1]
  try :
    return  cache_weekday[s2]
  except KeyError:
    wd= arrow.get(s2, fmt).weekday()
    cache_weekday[s2]= wd
  return wd

def season(d):
  m=  int(d[5:7])
  if m > 3 and m  < 10: return 1
  else: return 0 


def daytime(d):
  h= int(d[11:13])
  if   h < 11 :   return 0
  elif h < 14 : return 1    #lunch
  elif h < 18 : return 2    # afternoon
  elif h < 21 : return 3    # dinner
  else :        return 4   #night



coldate=  'purchased_at'   ; fmt= 'YYYY-MM-DD HH:mm:ss'
# df_v1['month']=   df_v1[coldate].apply(lambda t: arrow.get(t, fmt).month).astype('int8') 
df_v1['day']=     df_v1[coldate].apply(day).astype('int8') 
df_v1['month']=   df_v1[coldate].apply(month).astype('int8') 
df_v1['year']=    df_v1[coldate].apply(year).astype('int16') 
df_v1['hour']=    df_v1[coldate].apply(hour).astype('int8') 
df_v1['season']=  df_v1[coldate].apply(season).astype('int8') 
df_v1['daytime']= df_v1[coldate].apply(daytime).astype('int8') 
gc.collect()  


#--- Cache : Accelerate the compute Super Slow, dont forget the function weekday()  
df_v1['weekday']= df_v1[coldate].apply( weekday ).astype('int8') 
gc.collect()  


del df_v1['purchased_at']
df_v1.head()


df_v1.to_csv(out1+'/purchasings_v2.csv', index=False)
gc.collect()  





####### Pivot Table on the CSV file   ###############################################
df_pivot1= pd.pivot_table(df_v1,
  index='product_name' ,   columns='location_code', 
  values='amount', aggfunc='sum' )


df_pivot1.fillna(0, inplace=True)
df_pivot1


file1= 'df_pivot1_product_name_area_code_amount'
df_pivot1.to_csv(out1+ file1+'.csv', index=True)
util.save(df_pivot1, out1+file1 + '.pkl')
gc.collect()


df_pivot1['product_name'] = df_pivot1.index

df_pivot1.columns


#### Pandas Pivo


df_pivot1.to_csv(out1+ file1+'.csv', index=False)




###Location filtering  --------------------------------------------------------------




#### Pivot on items 






### Export to Excel






#####################################################################################
######Caching for date parsing



#### Pandas #########################################################################
df= pd.read_csv( rawcokeon + 'purchasings.csv', sep=',')   ##ll int

df_pivot1= pd.pivot_table(df, index='product_name', columns='bottler_code', 
                          values='price', aggfunc='sum' )


df_pivot1.fillna(0, inplace=True)
df_pivot1.to_csv(out1+'pivot_1.csv')
util.save(df_pivot1, 'df_pivot1_ref')


del df; gc.collect()
           
           
          
### DASK   ##########################################################################
df = dd.read_csv( rawcokeon + 'giving_tickets.csv', sep=',')

df.head(), len(df)

 
 
df = dd.read_csv( rawcokeon +  'purchasings.csv', sep=',')
df.head(), len(df)
gc.collect() 


df['bottler_code']= df['bottler_code'].astype('category')
df_pivot2= df.pivot_table(index='product_name', columns='bottler_code', 
                          values='price', aggfunc='sum' )
 
 
df_pivot2= df_pivot2.compute()
           
df_pivot2.fillna(0, inplace=True)    

df_pivot1.values - df_pivot2.values
###Works ok











################# Pandas SQL  to POSTGRES  ####################################################
import sqlalchemy as sql

# Start Postgres DB in pgadmin
# psycopg2
# dbengine = sql.create_engine('postgresql+psycopg2://postgres:postgres@localhost/coke')
dbengine= util.sql_create_dbengine('postgres',  dbname='coke', login='postgres', password='postgres', port=5432) 


df_q= util.sql_query(dbengine=dbengine, sqlr= ''' 
select * from purchasings where bottler_name='CCWJ' 


''')



util.pd_dtypes(df, returnasdict=0)








### ok works, no Encoding issues  !!!
df_q.to_sql('purchasings2', dbengine)





### Get SQL
####in PgAdmrawvm :
### http://www.vertabelo.com/blog/technical-articles/creating-pivot-tables-in-postgresql-using-the-crosstab-function
#CREATE extension tablefunc
'''
### Pivot table is NOT DYNAMIC !!!!!
Need to specify the columns  ---> Bad

'''

#####################################################################################
  




######################### CSV to HDFS ###############################################
print df_type_guess_all['purchasings']

dtype0= {'user_id': 'int32', 'temperature': 'int16', 'js_code': 'int32', 'bottler_code': 'int16', 
'area_code': 'int32', 'location_code': 'int32', 'machine_code': 'int64', 'price': 'int16', 
'amount': 'int16', 'id': 'int32', 'promotion_id': 'float16', 'purchased_at': 'object',
 'bottler_name': 'category', 'product_name': 'object'}

util.pd_h5_fromcsv_tohdfs(rawcokeon, 'purchasings.csv', tofilehdfs= rawcokeon + 'purchasings.h5',
                          chunksize= 2000000, tablename='df', mode='w',
                          dtype0=dtype0)
                  
                  
df2= pd.read_hdf(rawcokeon + 'purchasings.h5', 'df', start=0, stop=100)                      
df2.head(),  util.pd_info(df2)           
                  
                  
                          
               
                          


util.a_info_packagelist()


### DASK   ###############################################################
df = dd.read_csv(rawcokeon + 'user_groups.csv')




df = dd.read_hdf(rawcokeon + 'purchasings.h5', key='df', start=0, stop= 100)
df.head(),  df.dtypes
'nrow', len(df)

# df['company']= df['company'].astype('category')
df['purchaseamount']= df['purchaseamount'].astype('float32')
df['purchaseamount']= df['purchaseamount'].fillna(0.0)

# Left: index, Colums: top, values: center
df_pivot= df.pivot_table(index='chain', columns='dept', values='purchaseamount', aggfunc='sum')
df_pivotr= df_pivot.compute()
gc.collect()  #free memory
df_pivotr.shape


df['chain'].drop_duplicates().count().compute() #134
#


df['purchaseamount'].head()
df['purchaseamount'].sum().compute()


#############################################################################################


#Valid in DASK
df[df.resource_record!='AAAA'].resource_record.value_counts().compute()
df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})


####Saving DASK on disk
df.to_hdf('bigfile_merge.hdf', 'df',  compute=True)
gc.collect()  #free memory




################# Pandas SQL : Crashing with Unicode   #####################################################
from pandasql import *
import pandas as pd
pysqldf = lambda q: sqldf(q, globals())


#Clean to unicode
df= df.apply(da.str_to_unicode)     

           
q= """
SELECT m.product_name, m.js_code
FROM
  df m
WHERE
    m.bottler_code > 25;
"""        

###  Crash due to encoding issue
df_rs= pysqldf(q)           
           
           
df_rs= df[[ 'product_name', 'js_code' ]].where(df.bottler_code > 25 )
           

           
           











######################## DTYPES   ###############################################################
''' 'Object',  'bool', 'string_', 'unicode' float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64
'''
 
df= pd.read_csv(rawcokeon+'file_big01.csv', nrows=1000)
df.describe()
df_dtype0= util.pd_info(df) ; df_dtype0

dtype0= {'COUPON_ID_hash': 'object',
 'ITEM_COUNT': 'int16',
 'I_DATE': 'object',
 'PURCHASEID_hash': 'object',
 'SMALL_AREA_NAME': 'category',
 'USER_ID_hash': 'category'}

df= df.astype(dtype0)

######  To small int
for col in [ 'purchasequantity'  ] :
  df[col]=  df[col].astype('int32')


  

########## Japanese Text ############################################################
df= pd.read_csv('E:/_data/kaggle/couponjapan/rawcsv//coupon_list_test.csv')
df.dtypes

df2= df.astype(type_guess_all['coupon_list_test'])
df2.dtypes


#### unicode  --- > utf-8, before SAVING
df2['VALIDEND']= df2['VALIDEND'].apply(util.str_to_utf8)
df2['COUPON_ID_hash']= df2['COUPON_ID_hash'].apply(util.str_to_utf8)

df2.to_hdf('E:/_data/kaggle/couponjapan/rawcsv//coupon_list_test2.h5', 'df', format='table')





######################### CSV to HDFS ###############################################
data= 'E:/_data/kaggle/test/'
util.pd_h5_fromcsv_tohdfs(data, 'ja*.csv', tofilehdfs= data + 'fileja_hdf_compress3.h5',
                          chunksize= 10000, tablename='df', mode='w', dtypes=None, col_category=[])

store= pd.HDFStore(data + 'fileja_hdf_compress3.h5')
for df in store.select( 'df',  chunksize=5000)  :
   print df.shape,

###### BIG CSV
util.pd_h5_fromcsv_tohdfs(data, '*big*.csv', tofilehdfs= data + 'file_big_hdf_compress.h5',
                          chunksize= 100000, tablename='df', mode='a')


store= pd.HDFStore(data + 'file_big_hdf_compress.h5')
for df in store.select( 'df',  chunksize=500000) :
   print df.shape,



######### Operation on store    #####################################################
# http://pandas.pydata.org/pandas-docs/stable/io.html#query-via-data-columns
df.head()
df.index

store.get_storer('df').nrows


#Get Specific
df_col1= store.select('df', columns=['SMALL_AREA_NAME'])


#Select on range of Index
ix0 = store.select_as_coordinates('df','index> 5000')
ix0.summary()
store.select('df', columns=['SMALL_AREA_NAME'],   where=ix0)


#Select using a mask criteria
where = c[pd.DatetimeIndex(c).month==5].index
store.select('df_mask',where=where)

# Another way is to pass a PyTables query to the HDF file:
df= store.select('df', '(column1 == 1.0) & (column2 > 2.5) & (column3 < 10) & …)')





#####################################################################################
#####################################################################################
df = pd.DataFrame({'a':['1','2',3], 'b':['ホテ','・旅館', 'ホテル']})
df.dtypes


df= df.astype({'a': 'unicode', 'b': 'str'})
type(df.iloc[0, 0])


df_col= df.columns.values


df_type_len= [ df[col].map(len).max()  if str(df[col].dtype) == 'object' else 0  for col in df.columns.values ]


for col in df.columns.values :
  if str(df[col].dtype) == 'object'  :
    df[col].fillna('', inplace=True)
    print  df[col].map(len).max()

type(df.iloc[0,0])
df.iloc[0,0].decode('utf-8')


def pd_str_isascii(x):
  try :
    x.decode('ascii'); return True
  except: return false

   
def str_to_utf8(x):
  return x.encode('utf-8')


#####################################################################################
df['SMALL_AREA_NAME']= df['SMALL_AREA_NAME'].apply(to_unicode)

util.pd_toexcel(df, outfile=rawcokeon+'test.xlsx')



2. Unicode everywhere
to_unicode_or_bust(ivan_uni)


3. Encode late
Encode to <type 'str'> when you write to disk or print

f = open('/tmp/ivan_out.txt','w')
f.write(ivan_uni.encode('utf-8'))
f.close()


np.finfo(np.float16).max

da.csv_analysis()
df1.dtypes.tolist()

type(df1.iloc[0,1])









#------ Parse CSV Files and put Summary into Excel   --------------------------------
da.csv_col_schema_toexcel(dircsv=indir0 + '/rawcsv', outfile=indir0 + '/meta_summary2.xlsx', returntable=1,
                          maxcol_pertable=90, maxstrlen='U80')


#------   Need to find Type:
 int, float, string lenght, min length



#########################  Pivot Table #########################################################################
df_pivot1= da.csv_pivotable(fileh5=data+'fileja_hdf_compress3.h5',
                            leftX='I_DATE', topY='SMALL_AREA_NAME', centerZ='ITEM_COUNT', mapreduce='sum',
                            chunksize= 50000, tablename='df')
df_pivot1

store.get_storer('df').nrows



### 1.2 Go Big table :  OK !!
df_pivot1= da.csv_pivotable(fileh5=data+'/file_hdf_table_compress.h5',
leftX='chain', topY='company', centerZ='purchasequantity', mapreduce='sum', chunksize= 800000, tablename='df')
df_pivot1


util.save(df_pivot1, data+'/df_pivot1.pkl')



######## On Disk Storage #######################################################################

#### pkl file:  No issue, 1.7go --->  1.3 go      OK validate
util.py_save_obj(df, data+'/transaction_red.pkl')

df= util.py_load_obj(data+'/transaction_red.pkl', isabsolutpath=1)


#### hdf:  Low size +  Super Fast
df.to_hdf(data+'transact_hdf.h5','df',mode='w',format='table')


#### hdf:  Compress 0.7Go
# def to_hdf(self, path_or_buf, key, mode='a', append=False, get=None, **kwargs)
df.to_hdf(data+'file_hdf_table_compress.h5','df',mode='w',format='table', complib='blosc')


#### Read needs 6Go in memory....., can sub-select
df= pd.read_hdf(data+'file_hdf_table_compress.h5','df', start=75000, stop= 60000)
df.info()


##################################################################################################
#################### Japanese Text ###############################################################
df= pd.read_csv('E:/_data/kaggle/couponjapan/rawcsv/coupon_detail_train.csv')
df.info()

#### cannot reduce type


#### hdf:  23Mo --> 17Mo  compress
# def to_hdf(self, path_or_buf, key, mode='a', append=False, get=None, **kwargs)
df.to_hdf(data+'japa_hdf_table_compress.h5','df',mode='w',format='table', complib='blosc')


#### Read needs 6Go in memory....., can sub-select
df= pd.read_hdf(data+'japa_hdf_table_compress.h5','df', start=75, stop= 6000)
df.info()
##################################################################################################





############# POSGRES Database      ###################################################################
odo(df, 'postgresql://localhost:5432::transact')  # Migrate dataframe to Postgres

odo('myfile.*.csv', 'postgresql://hostname::tablename')  # Load CSVs to Postgres

psycopg2
conda install -c conda-forge psycopg2=2.6.2






################## Initial Load and type check Max Value    ###############################################
df1= pd.read_csv(file_transact, sep=',', nrows= 100)
dfs= df1.describe()
print df1.head(10),"\n",  "\n\n",  util.pd_dtype_print(df1)
dfs[dfs.index.isin(['min', 'max'])]


ds= {'id': 'uint32', 'chain': 'uint16', 'dept': 'uint16', 'category': 'uint32', 'company': 'uint32',
'brand': 'uint32', 'date': 'object', 'productsize': 'float16', 'productmeasure': '|S2',
'purchasequantity': 'uint16', 'purchaseamount': 'float16'}


DIRRAW= ''
DBNAME + TABLE1= ''


DIR0 = '  '
df1= pd.read_csv(f_transactions_red.csv, sep=',', nrows= 100)


file1= data + 'trainHistory.csv'
file2= data + 'transactions_red.csv'
file3= data + 'transactions_red2.csv'

df_bz= bz.data( file2)
df_bz.dshape

pivot1= bz.by( df_bz.chain ,  col1=  df_bz.purchaseamount.sum() )


df= odo(file3, pd.DataFrame)
df.dtypes



by(merge(df.name, df.id), amount=df.amount.mean())

#### SQL to Blaze
http://blaze.readthedocs.io/en/latest/rosetta-sql.html




import dask
df= dask.dataframe.csv.read_csv(file3)



 by(table.grouping_columns, name_1=table.column.reduction(),
                         name_2=table.column.reduction(),)
Here is a concrete example. Find the shortest, longest, and average petal length by species.

 from blaze import by
 by(iris.species, shortest=iris.petal_length.min(),
                  longest=iris.petal_length.max(),
                average=iris.petal_length.mean())




##### Load 2.6Go in RAM
df= pd.read_csv(data + '/transactions_red.csv')



df_pivot1= da.csv_pivotable(fileh5=data+'japa_hdf_table_compress2.h5',
                            leftX='I_DATE', topY='SMALL_AREA_NAME', centerZ='ITEM_COUNT', mapreduce='sum',
                            chunksize= 5000, tablename='df')


df_pivot1.info()
df_pivot1.head()


store= pd.get_store(data+'japa_hdf_table_compress.h5')

pivotable1 = reduce(lambda x, y: x.add(y, fill_value=0), (df.groupby(by='SMALL_AREA_NAME').sum() for df in store.select(tablename, chunksize=chunksize)))


df_pivot= df.pivot(index='I_DATE', columns='SMALL_AREA_NAME', values='ITEM_COUNT')

I_DATE,SMALL_AREA_NAME,


pivot_i = pd.DataFrame.pivot_table(df, values='ITEM_COUNT', index=['I_DATE'],
                                 columns=['SMALL_AREA_NAME'], aggfunc=np.sum,
                                 fill_value=0)

pivot0= pd.concat([pivot0, pivot_i]).groupby(by=['I_DATE']).sum()




table.info()
df_list = [table, table]

pd.concat([x1, x2]).groupby(["city", "school"], as_index=False)["count"].sum()

table.index


df_pivot1= da.csv_pivotable(dircsv='', filepattern='', fileh5=data+'japa_hdf_table_compress.h5',
                            leftX='chain', topY='market', mapreduce='sum', chunksize= 500000, tablename='df')


'''
http://stackoverflow.com/questions/32298047/efficient-storage-of-large-string-column-in-pandas-dataframe

df.to_hdf('test_compression_table.h5','df',mode='w',format='table',complib='blosc')

df['A'] = df['A'].astype('category')

df.to_hdf('test_categorical_table.h5','df',mode='w',format='table')

'''


util.pd_dtype_print(df2)



########## 1.7Go CSV file   ######################################################################
util.pd_h5_fromcsv_tohdfs(data, filepattern='transactions_red.csv', tofilehdfs=data+'file2.h5', tablename='df',  chunksize= 1000000)


df= util.pd_h5_load(data+'/file2.h5', 'df', rowstart=0, rowend=10)



util.pd_h5_dumpinfo(data+'/file2.h5')


df_pivot1= da.csv_pivotable(dircsv=data, filepattern='coupon_detail_train.csv', fileh5=data+'/transact01.h5',
                            leftX='chain', topY='market', mapreduce='sum', chunksize= 500000, tablename='df')




id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount
86246,205,99,9909,104538848,15343,2012-03-02,16,OZ,1,2.49
86246,205,21,2106,105100050,27873,2012-03-02,64,OZ,1,3.29




###########################################################################################






util.pd_h5_fromcsv_tohdfs(data, filepattern='*.csv', tofilehdfs=data+'file1.h5', tablename='data', encoding='utf-8', chunksize= 2000000)


util.os_file_listall(data, pattern='*train*.csv')[2]



df[columns] = df[columns].applymap(str)



format='table', min_itemsize={'A': 30}


'''
# import the modules
import unicodedata as ucd
import pandas as pd

# open the data file
df = pd.read_csv('path/to/file.csv', encoding='utf-8')

# the column we want to normalize is named 'page'
df['page'] = df['page'].map(lambda x: ucd.normalize('NFKD', x))


'''

filelist1= util.os_file_listall(dircsv=data, pattern='*train*.csv')

filelist1[2]

csvlist=   filelist1[:,0]


id,chain,offer,market,repeattrips,repeater,offerdate


id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount





#############################################################################################
########################### Blaze Load  #####################################################

df = bz.data(file1)
print df.peek()
df.count()

df.dshape

ds= od.dshape("""var * {
  id: int64,
  chain: int32,
  dept: int32,
  category: int64,
  company: int64,
  brand: int64,
  date:  string[30, 'ascii'],
  productsize: float32,
  productmeasure: string[10, 'ascii'],
  purchasequantity: int32,
  purchaseamount: float32
  }""")

df = bz.data(file1, dshape=ds)

#Selection with Blaze
df2= df[  df.month  > 10]
df2= df2[ df2.dateyear  > 2015 ]

df2.head(10)
df_pd=df2.head(10)
df_pd= odo(df_pd, pd.DataFrame)


bz.mean(df2.close)

#Export to DataFrame the sub-selection
df_pd= odo(df2, pd.DataFrame)

#Export to Numpy
df_np=  df_pd[['close']].values
len(df_np)
#############################################################################################





















#-------------- Details:  -----------------------------------------
util.session_load('/kaggle/kaggle_supermarket_01')

util.a_help()



##########################################################################################
#Kaggle Challenge:
#'Reduce the data and generate features' by Triskelion
# 10 mio:  1.5Go, 30mio: 4.6Go


#########################################################################################
#########################  Data Selection ###############################################
data= 'E:/_data/kaggle/largesuper/csv/'
out1= 'E:/_data/kaggle/out/'

file1= data+'*.csv'  #Full

file1= data+'transactions.csv.gz'

loc_offers =   data+ "offers.csv"
file_transact= data+ "transactions_red.csv"
file_reduced =  out1+ "/transac_red.csv" # will be created


'''
0   1     2     3         4      5    6      7              8               9                10
id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount
86246,205,7,707,1078778070,12564,2012-03-02,12,OZ,1,7.59
86246,205,63,6319,107654575,17876,2012-03-02,64,OZ,1,1.59
86246,205,97,9753,1022027929,0,2012-03-02,1,CT,1,5.99
'''

###### Need to find type in



################## Initial Load and type check Max Value    ##########################################
df1= pd.read_csv(file_transact, sep=',', nrows= 100)
dfs= df1.describe()
print df1.head(10),"\n",  "\n\n",  util.pd_dtype_print(df1)
dfs[dfs.index.isin(['min', 'max'])]


ds= {'id': 'uint32', 'chain': 'uint16', 'dept': 'uint16', 'category': 'uint32', 'company': 'uint32',
'brand': 'uint32', 'date': 'object', 'productsize': 'float16', 'productmeasure': '|S2',
'purchasequantity': 'uint16', 'purchaseamount': 'float16'}


#Check
df1= pd.read_csv(file_transact, sep=',', nrows= 10, dtype=ds)
print util.pd_dtype_print(df1), '\n', df1.head(3)



#Full Load, 10mio: 1.5Go, Remove Columns
df1= pd.read_csv(file_transact, sep=',', nrows= 1*10**6, usecols=np.arange(1,9), dtype=ds)
print df1.describe(),"\n",  util.pd_info_memsize(df1)

# util.session_save('kaggle_supermarket02', globals())


################## Get  Category distribution   ##################################################
catdict  = da.csv_col_get_dictfreq(file_transact, category_cols= [1, 2, 3, 4, 5, 7, 8], maxline=1 * 10 ** 6)
util.py_save_obj(catdict, '/kaggle/catdict')
# 'D:\\_devs\\Python01\\project27/aaserialize/catdict.pkl'


catfreq= da.col_categorydict_freqstudy(catdict)
# util.py_save_obj(catfreq, '/kaggle/catfreq')



catfreq= util.py_load_obj('/kaggle/catfreq')


# Generate the categories values to be kept   -----------------------------------------------------
file_category=  data+ "offers.csv"
ncol= 8   #nb of Column than in Transaction File
catval_tokeep=[ {} for i in xrange(0, ncol)]
for i, line in enumerate(open(file_category)):
    ll=  line.split(",")
    catval_tokeep[3][  ll[1] ]  = 1  # Offer_file_col1 --> Transact_file_col_3
    catval_tokeep[4][  ll[3] ] =  1  # Offer_file_col3 --> Transact_file_col_4

catval_tokeep


#Filter function to select row based on some pre-filled category / conditions---------------------
def condfilter(colk, catval_tokeep) :
  if colk[3] in catval_tokeep[3] or colk[4] in catval_tokeep[4]: return True
  else: return False


da.csv_row_reduce_line(fromfile=file_transact, tofile=file_reduced, condfilter=condfilter,
                       catval_tokeep=catval_tokeep, maxline=100)


infile2=  file_reduced


#------Features Generation from data   ---------------------------------------------------------
util.session_save('/kaggle/supermarket_01', globals())


catval_tokeep2= util.np_dict_tolist(catval_tokeep)




### Add Date  ----------------------------------------------------------------------
import arrow
### arrow is super slow

def day(t):   return arrow.get(t, 'YYYY-MM-DD HH:mm:ss').day
def season(d):
  m= arrow.get(d, 'YYYY-MM-DD HH:mm:ss').month
  if m > 3 and m  < 10 : return 1 
  else : return 0  

def daytime(d):
  h= arrow.get(d, 'YYYY-MM-DD HH:mm:ss').hour
  if h < 11 :   return 0
  elif h < 14 : return 1    #lunch
  elif h < 18 : return 2    # afternoon
  elif h < 21 : return 3    # dinner
  else :        return 4   #night

coldate=  'purchased_at'   ; fmt= 'YYYY-MM-DD HH:mm:ss'
df_v1['day']=     df_v1[coldate].apply(day).astype('int8') 
df_v1['month']=   df_v1[coldate].apply(lambda t: arrow.get(t, fmt).month).astype('int8') 
df_v1['year']=    df_v1[coldate].apply(lambda t: arrow.get(t, fmt).year).astype('int16') 
df_v1['hour']=    df_v1[coldate].apply(lambda t: arrow.get(t, fmt).hour).astype('int8') 
df_v1['weekday']= df_v1[coldate].apply(lambda t: arrow.get(t, fmt).weekday).astype('int8') 
df_v1['season']=  df_v1[coldate].apply(season).astype('int8') 
df_v1['daytime']= df_v1[coldate].apply(daytime).astype('int8') 


df_v1.to_csv(out1+'/purchasings_v1.csv')









##################################################################################################
'''
Batch mode to re-collect the data
SQL batch on 1 million line
 Year / Month  /  LocationType  / DrinkType /Brand

Flat Files

10 million DAY


MachineID, DrinkID

pivot

http://jihan.30maps.com/map/67061


'''


''' Map Reduce
product ---> Soda / Coffee / Water   Hot / Cold

Machines ---> LocationType         Conbini /
(Onsen area, bath, Sport area, Functionnal, ...)


Seasonanlity ---> Split in Time


SQL Data


From one Place ---> Need to find amenuiities within 50m

'''


util.session_save("/kaggle/kaggle_supermarket_01")


# Distribition

# pandas Method for First Analysis :
df.describe(), df1.hist()


#Histogram by Class
df1.groupby('company').hist()


# Feature vs Feature Distribution / Col vs Col
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
'''
# 25s for 1m line

'''
#--------------------------
SALES.groupby('name')['quantity'].sum().plot(kind="bar")

'''
# Accidents which happened on a Sunday, > 20 cars
rs = df[ (df.Day_of_Week == 1) & (df.Number_of_Vehicles > 20)]

# Convert date to Pandas date/time
london_data_2000 = data[
    (pd.to_datetime(data['Date'], coerce=True) >
        pd.to_datetime('2000-01-01', coerce=True)) &
    (pd.to_datetime(data['Date'], coerce=True) <
        pd.to_datetime('2000-12-31', coerce=True))
]
'''


data_csv_mapreduce(loc_offers, loc_transactions, loc_reduced)






################################  IN MEMORY Structure #############################################
# Info Type :
util.pd_info_memsize(df2)

#Convert to pandas dataframe
df_pd= odo(df, pd.DataFrame)

util.pd_info_memsize(df_pd)



#------- Data Clean Up Before Saving -------------------------------------------------------------
df_pd2=  util.pd_dtypes_type1_totype2(df_pd, fromtype=np.datetime64, targetype=str)

df_pd2=  util.pd_dtypes_type1_totype2(df_pd, fromtype=np.int64, targetype=np.int32)

df_pd2= util.pd_h5_cleanbeforesave(df_pd2)


util.pd_h5_save(df_pd2, 'F:/_data/stock/intraday/q5min/usetf1.h5', 'data')


df3= util.pd_h5_load('F:/_data/stock/intraday/q5min/usetf1.h5', 'data')

util.pd_info_memsize(df3)

df.dshape



################### transformation Columns ######################################################
'''
transformation Columns :

df[['col2','col3']] = df[['col2','col3']].apply(pd.to_numeric)

df.apply(lambda x: pd.to_numeric(x, errors='ignore'))


'''


dir1= 'E:/_data/kaggle/largesuper/csv/'
df= odo(dir1 + '*train*.csv', pd.DataFrame) # Stream through many CSV files

odo(dir1 + '*train*.csv', 'hdfstore://'+dir1 +'file.h5::/df/')



################################ Saving File on DISK #############################################
#  Show data Type
pd.discover(df),  df.dshape

# ODO Conversion possibilities
 odo(df, list)  # create new list from Pandas DataFrame
 odo(df, [])  # append onto existing list
 odo(df, 'myfile.json')  # Dump dataframe to line-delimited JSON
 odo('myfiles.*.csv', Iterator) # Stream through many CSV files
 odo(df, 'postgresql://hostname::tablename')  # Migrate dataframe to Postgres
 odo('myfile.*.csv', 'postgresql://hostname::tablename')  # Load CSVs to Postgres
 odo('postgresql://hostname::tablename', 'myfile.json') # Dump Postgres to JSON
 odo('mongodb://hostname/db::collection', pd.DataFrame) # Dump Mongo to DataFrame


# To ensure that you encode your dataset appropriately we recommend passing a datashape explicitly.
# As in our previous example this can often be done by editing automatically generated datashapes
# Transpose into Datashape  Before Converting to Flat File
# ds = dshape("var * {name: string[20, 'ascii'], amount: float64}")
d.dshape

ds= od.dshape("""var * {
  date: string[30, 'unicode'],
  open: float64, high: float64, low: float64, close: float64, volume: int64,
  symbol: string[6, 'unicode'],
  dateyear: int32, month: int32,day: int32,   hour: int32,  minute: int32
  }""")

#Size of Datetime : len(str( df_pd['date'].values[0]  ))


#  Convert to Format
odo(df_pd, 'F:/_data/stock/intraday/q5min/usetf2.bcolz', dshape=ds)


#Test of Loading it
df2= bz.data('F:/_data/stock/intraday/q5min/usetf1.bcolz')
df2.head(5),  bz.mean(df2.close






# BCOLZ :  Only for Numerical / Need Datashape to reformat data
odo(file1, 'F:/_data/stock/intraday/q5min/q5min_etf.bcolz')


#NO NO ODO HDFS: ODO is bad   : Need to Clean Up the data, BETTER TO Pandas Directly
odo(df2, 'hdfstore://F:/_data/stock/intraday/q5min/usetf9.h5::/data/')

odo(df_pd[['open','close', 'symbol']], 'hdfstore://F:/_data/stock/intraday/q5min/usetf2.h5::/data/')


#Pandas HDFS: Need to Clean Up the data, BETTER TO Pandas Directly, Release Lock
df_pd= util.pd_hdfs_cleanbeforesave(df_pd)

util.pd_h5_save(df_pd2[['open','sym']], 'F:/_data/stock/intraday/q5min/usetf11.h5', 'data')

df3= util.pd_h5_load('F:/_data/stock/intraday/q5min/usetf9.h5', 'data')


# into  SINGLE CSV
odo(df_pd, 'F:/_data/stock/intraday/q5min/usetf.csv')


#PKL on Pandas table
util.py_save_obj(df_pd, 'F:/_data/stock/intraday/q5min/usetf.pkl', otherfolder=1)


# SQLite  Database


# Job Lib to save Numpy array : Better
# Joblib, with 2 different versions, 0.9.4 and master (dev),




'''

#NO NO  ODO  Pytable NO NOcannot support date  NEVER USE
odo(df3, 'F:/_data/stock/intraday/q5min/usetf.h5::/data/')

#NO NO :   NUMPY MEMMAP : NO only 2.5go
    generators (e.g. to read a file a line at a time)
    Key-value stores (e.g. redis)
    SQL and NoSQL databases (e.g. sqlite3)
'''


###########################################################################################




#Unicode Issues with HFS
types = df_pd.apply(lambda x: pd.lib.infer_dtype(df_pd.values))

types[types=='unicode']

for col in types[types=='unicode'].index:
   # df_pd[col] = df_pd[col].astype(str)
   df[col] = df[col].apply(lambda x: x.encode('utf-8').strip())






#  Expression Interactive :

db.iris.species.<tab
db.iris.species.columns       db.iris.species.max
db.iris.species.count         db.iris.species.min
db.iris.species.count_values  db.iris.species.ndim
db.iris.species.distinct      db.iris.species.nunique
db.iris.species.dshape        db.iris.species.relabel
db.iris.species.expr          db.iris.species.resources
db.iris.species.fields        db.iris.species.schema
db.iris.species.head          db.iris.species.shape
db.iris.species.isidentical   db.iris.species.sort
db.iris.species.label         db.iris.species.species
db.iris.species.like          db.iris.species.to_html
db.iris.species.map











#Cannot Compute any expression in Blaze

from dateutil.parser import parse
dateparse= lambda x: parse(x)





#  Cannot apply
def colfun(x) : lambda  : x+5
   x+5


bz.transform(df, dateday = util.datestring_todatetime(df.date) )

datetimes = t.time.map(datetime.utcfromtimestamp,     schema='{time: datetime}')


df[0,5].values

df.amount.map(lambda x: x + 1,  'int64')

df_pd= df[1:500]

df_pd.count()


bz.mean(df.close)



df2= df.apply(colfun, dshape='float32')  # Hash value of resultant dataset


#  Data
http://blaze.pydata.org/blog/2015/09/08/reddit-comments/

http://blaze.pydata.org/blog/2015/09/08/reddit-comments/


# Type Conversion
Type conversion of expressions can be done with the coerce expression. Here’s how to compute the average account balance
for all the deadbeats in my accounts table and then cast the result to a 64-bit integer:


 deadbeats = accounts[accounts.balance < 0]
 avg_deliquency = deadbeats.balance.mean()
 chopped = avg_deliquency.coerce(to='int64')
 chopped.mean(accounts[accounts.balance < 0].balance).coerce(to='int64')


############## Cost      ########################
http://odo.pydata.org/en/latest/aws.html



#####################################################################################
#########################  LOAD Data  ###############################################
df = bz.data('sqlite:///%s::iris' % bz.utils.example('iris.db'))

df = bz.data('my-small-file.csv')

df = bz.data('myfile-2014-01-*.csv.gz')

HDFS  ::/datapath

engine = sql.create_engine('postgresql://%s:%s@localhost:5432/%s' %(myusername, mypassword, mydatabase))
df     = bz.data(engine)

engine = sql.create_engine('postgresql://myusername:mypassword@localhost:5432/mydatabase' %(myusername, mypassword, mydatabase))



#Show the data
df.peek()




#----------------------Server     ---------------------------------------------------
from numba import vectorize, float64
from numpy import linspace, pi
from blaze import Data, discover, sqrt, exp
x = Data(linspace(-5, 5, 100000000))
mu, sigma = -1.33, 1.25
expr = 1 / (sigma * sqrt(2 * pi)) * exp(-(x - mu) ** 2 / (2 * sigma ** 2))






#####################################################################################
#########################  CLEAN   ##################################################
df.relabel(name='alias',  amount='dollars')

#Remove Duplicate
df.distinct()
df.name.distinct()


df= df[df.amount < 0]

#####################################################################################
#########################  MAP REDUCE  ##############################################
# Group by and Map-Reduce operations
bz.by(df.species, minimum=df.petal_length.min(), maximum=df.petal_length.max(),
                  ratio= df.petal_length.max() - df.petal_length.min())


df['petal_length'].count()

df['petal_length'].map(lambda x: x + 1,'int64')


bz.transform(df, sepal_ratio = df.sepal_length / df.sepal_width,
                 petal_ratio = df.petal_length / df.petal_width)


df.petal_length.map(lambda x: x + 1,'float64')



#Find Colmun= text
df[df.species.like('*versicolor')]


# Add new Columns : Add new columns using the transform function

transform(iris, sepal_ratio = iris.sepal_length / iris.sepal_width,
                petal_ratio = iris.petal_length / iris.petal_width)


#Convert file to dataframe
odo((1, 2, 3), pd.DataFrame)




#####################################################################################
#########################  LOAD Data  ###############################################
df = bz.data('sqlite:///%s::iris' % bz.utils.example('iris.db'))

df = bz.data('my-small-file.csv')

df = bz.data('myfile-2014-01-*.csv.gz')

#Show the data
df.peek()




Basic Queries
Here we give a quick overview of some of the more common query functionality.

We use the well known iris dataset

 from blaze import data
 from blaze.utils import example
 iris = data(example('iris.csv'))
 iris.peek()
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
...
Column Access
Select individual columns using attributes

 iris.species
        species
0   Iris-setosa
1   Iris-setosa
2   Iris-setosa
3   Iris-setosa
...
Or item access

 iris['species']
        species
0   Iris-setosa
1   Iris-setosa
2   Iris-setosa
3   Iris-setosa
...
Select many columns using a list of names

 iris[['sepal_length', 'species']]
    sepal_length      species
0            5.1  Iris-setosa
1            4.9  Iris-setosa
2            4.7  Iris-setosa
3            4.6  Iris-setosa
...
Mathematical operations
Use mathematical operators and functions as normal

 from blaze import log
 log(iris.sepal_length * 10)
    sepal_length
0       3.931826
1       3.891820
2       3.850148
3       3.828641
...
Note that mathematical functions like log should be imported from blaze. These will translate to np.log, math.log, sqlalchemy.sql.func.log, etc. based on the backend.

Reductions
As with many Blaze operations reductions like sum and mean may be used either as methods or as base functions.

 iris.sepal_length.mean()
5.84333333333333...

 from blaze import mean
 mean(iris.sepal_length)
5.84333333333333...
Split-Apply-Combine
The by operation expresses split-apply-combine computations. It has the general format

 by(table.grouping_columns, name_1=table.column.reduction(),
...                            name_2=table.column.reduction(),
...                            ...)
Here is a concrete example. Find the shortest, longest, and average petal length by species.

 from blaze import by
 by(iris.species, shortest=iris.petal_length.min(),
...                   longest=iris.petal_length.max(),
...                   average=iris.petal_length.mean())
           species  average  longest  shortest
0      Iris-setosa    1.462      1.9       1.0
1  Iris-versicolor    4.260      5.1       3.0
2   Iris-virginica    5.552      6.9       4.5
This simple model can be extended to include more complex groupers and more complex reduction expressions.

Add Computed Columns
Add new columns using the transform function

 transform(iris, sepal_ratio = iris.sepal_length / iris.sepal_width,
...                 petal_ratio = iris.petal_length / iris.petal_width)
    sepal_length  sepal_width  petal_length  petal_width      species  \
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa

    sepal_ratio  petal_ratio
0      1.457143     7.000000
1      1.633333     7.000000
2      1.468750     6.500000
3      1.483871     7.500000
...
Text Matching
Match text with glob strings, specifying columns with keyword arguments.

 iris[iris.species.like('*versicolor')]
    sepal_length  sepal_width  petal_length  petal_width          species
50           7.0          3.2           4.7          1.4  Iris-versicolor
51           6.4          3.2           4.5          1.5  Iris-versicolor
52           6.9          3.1           4.9          1.5  Iris-versicolor
Relabel Column names
 iris.relabel(petal_length='PETAL-LENGTH', petal_width='PETAL-WIDTH')
    sepal_length  sepal_width  PETAL-LENGTH  PETAL-WIDTH      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
Examples
Blaze can help solve many common problems that data analysts and scientists encounter. Here are a few examples of common issues that can be solved using blaze.

Combining separate, gzipped csv files.
 from blaze import odo
 from pandas import DataFrame
 odo(example('accounts_*.csv.gz'), DataFrame)
   id      name  amount
0   1     Alice     100
1   2       Bob     200
2   3   Charlie     300
3   4       Dan     400
4   5     Edith     500
Split-Apply-Combine
 from blaze import data, by
 t = data('sqlite:///%s::iris' % example('iris.db'))
 t.peek()
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
...
 by(t.species, max=t.petal_length.max(), min=t.petal_length.min())
           species  max  min
0      Iris-setosa  1.9  1.0
1  Iris-versicolor  5.1  3.0
2   Iris-virginica  6.9  4.5





#############################################################################################################
#############################################################################################################
#------ Output Directory    ####################################################################
dir2='E:/_data/stock/intraday/raw/'
now1= util.date_now()


#------------------Download 5min US ETF ########################################################
sym= alldata.us_etf_all_0
inter_sec= 300; nday=6000
for symbol in sym:
  pf.imp_googleIntradayQuoteSave(symbol,  inter_sec, nday, dircsv= dir2 +  'usetf_' + now1  )



#------------------Download 5min US Stock- #####################################################
sym= alldata.us_sp2000
inter_sec= 300; nday=6000
for symbol in sym:
  pf.imp_googleIntradayQuoteSave(symbol,  inter_sec, nday, dircsv= dir2 + 'ustock_' + now1 )





#------------------ Backup database  #####################################################
now1= util.date_now()
util.os_zipfolder('E:/_data/stock/intraday/q5min/',  'E:/_data/stock/intraday/q5min_'+now1 +'.zip',
                  iscompress= True)



#------------------ update Master  DB ###################################################
indir0=   'E:/_data/stock/intraday/raw/'
dirlist=    util.os_file_listall(indir0, '*', 1, onlyfolder=1)[0]

for dir0 in dirlist :
  dirname=  dir0[:dir0.find('_')]
  indir1= indir0 + dir0 +'/'
  file1= util.os_file_listall(indir1, '*.csv', 0)[:,0]
  file1= np.array([  x+'.csv' for  x in   file1  ])

  print dir0, dirname, len(file1)

  imp_db_csv_update(indir=indir1,
                      outdir='E:/_data/stock/intraday/q5min/'+dirname+'/', filelist=file1, intype='csv',
                      refcols=['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'])

#############################################################################################################
#############################################################################################################






#   Select Column not in the Dataframe




raise KeyError('%s not in index' % objarr[mask])
KeyError: "['date' 'open' 'high' 'low' 'close' 'volume' 'symbol'] not in index"



'''
indir1= indir0 + 'usetf_20161208/'
file1= util.os_file_listall(indir1, '*.csv', 0)[:,0]
file1= np.array([  x+'.csv' for  x in   file1  ])


df= imp_csv_dbupdate(indir=indir1,
                      outdir='E:/_data/stock/intraday/q5min/usetf/', filelist=file1, intype='csv',
                      refcols=['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'])



#------------------ update Master STOCK  ##############################################
indir1= 'E:/_data/stock/intraday/raw/' + 'ustock_20161208/'

file1= util.os_file_listall(indir1, '*.csv', 0)[:,0]
file1= np.array([  x+'.csv' for  x in   file1  ])


df= imp_csv_dbupdate(indir=indir1,
                      outdir='E:/_data/stock/intraday/q5min/ustock/', filelist=file1, intype='csv',
                      refcols=['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'])





df= imp_csv_dbupdate(indir=indir1,
                      outdir='E:/_data/stock/intraday/q5min/us/etf/', filelist=file1, intype='csv',
                      refcols=['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'])




df





#------------------ Put into Database -------------------------------------------



import odo

odo('sqlite:///db.db::q5min_etf', )






import bcolz


file1= 'E:/_data/stock/intraday/20161207_etf/NKE_20160928_300_6000.csv'
df= pf.imp_csv_toext(file1= file1,  sym='NKE', header=0,
cols= ['date', 'symbol','open','high','low','close','volume'], date=[1],
fromzone='Japan', tozone='UTC')



dircsv= 'E:/_data/stock/intraday/20161115_etf/'
df= pf.imp_hdfs_db_updatefromcsv(dircsv,filepd= r'E:/_data/stock/intraday/q5min/us/etf/',
                                 fromtimezone='Japan', tozone='UTC')



DIRCWD= 'E:/_data/stock/intraday/intraday_google_usetf2.h5'
df= pf.imp_hdfs_getquote(DIRCWD, 'SPY')
df1 = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]




df2= pf.imp_csv_toext(file1='E:/_data/stock/intraday/q5min/us/etf/SPY.csv',
      fromzone='Japan', tozone='UTC', header=0,
      cols=['date', 'open','high','low','close','volume', 'symbol'])

df2.date.values[0]










sym1= np.array(['AAXJ','ACWI','ACWV','ACWX','AGG','AGZ','AMJ','AMLP','AMU','AOM','ASHR','ATMP','BAB','BBH','BIL','BIV','BKLN','BLV','BND','BNDS','BNDX','BNO','BOND','BSCG','BSCH','BSCI','BSCJ','BSCK','BSCM','BSJH','BSJI','BSJJ','BSV','BWX','CHAD','CIU','CLY','CMF','CORN','CORP','CSJ','CSM','CWB','CWI','DBA','DBAW','DBB','DBC','DBEF','DBEM','DBEU','DBJP','DBKO','DBO','DBP','DEM','DES','DFE','DFJ','DGL','DGRO','DGRW','DHS','DIA','DIV','DJP','DLN','DLS','DNO','DOG','DON','DSI','DTN','DTO','DVY','DWM','DWX','DXJ','DZZ','ECH','EDV','EELV','EEM','EEMA','EEMS','EEMV','EFA','EFAV','EFG','EFV','EFZ','EIDO','EIS','EMB','EMLC','EMLP','EPHE','EPI','EPOL','EPP','EPU','ERUS','EUFN','EUM','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWN','EWP','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','EXI','EZA','EZU','FBT','FCG','FCOM','FDL','FDN','FDT','FENY','FEX','FEZ','FHLC','FIDU','FLOT','FM','FNCL','FNDA','FNDF','FNDX','FNX','FPE','FPX','FSTA','FTA','FTC','FTSD','FTSL','FTSM','FUTY','FV','FVD','FXA','FXB','FXC','FXD','FXE','FXG','FXH','FXI','FXL','FXO','FXU','FXY','FYX','GDX','GDXJ','GGOV','GLD','GMF','GMM','GNR','GOVT','GREK','GSG','GSLC','GSY','GUNR','GVI','GWX','GXC','HACK','HDG','HDGE','HDV','HEDJ','HEFA','HEWG','HEWJ','HEZU','HYD','HYEM','HYG','HYLS','HYMB','HYS','IAT','IAU','IBB','ICF','IDLV','IDU','IDV','IEF','IEFA','IEI','IEMG','IEO','IEUR','IEV','IEZ','IFGL','IGE','IGF','IGOV','IGV','IHDG','IHE','IHF','IHI','IJH','IJJ','IJK','IJR','IJS','IJT','ILF','INDA','INDY','ISTB','ITA','ITB','ITE','ITM','ITOT','ITR','IUSG','IUSV','IVE','IVV','IVW','IWB','IWC','IWD','IWF','IWM','IWN','IWO','IWP','IWR','IWS','IWV','IWY','IXC','IXJ','IXP','IXUS','IYC','IYE','IYF','IYG','IYH','IYJ','IYK','IYM','IYR','IYT','IYW','IYY','IYZ','JJC','JKE','JNK','KBE','KBWB','KIE','KRE','KWEB','KXI','LEMB','LQD','LTPZ','MBB','MCHI','MDY','MDYG','MDYV','MGC','MGK','MGV','MINT','MLPI','MLPN','MOO','MTUM','MUB','NEAR','NIB','NOBL','NTG','OEF','OIH','OIL','ONEQ','OUSA','PALL','PBP','PCY','PDP','PEJ','PEY','PFF','PGF','PGX','PHB','PHDG','PHYS','PICK','PID','PIE','PIN','PJP','PKW','PPH','PPLT','PRF','PSL','PSP','PSQ','PWB','PWV','PXH','PZA','QAI','QDEF','QDF','QQEW','QQQ','QUAL','REGL','REM','RFG','RGI','RHS','RIGS','RLY','RPG','RPV','RSP','RSX','RWO','RWR','RWX','RYH','RYT','RYU','SBIO','SCHA','SCHB','SCHC','SCHD','SCHE','SCHF','SCHG','SCHH','SCHM','SCHO','SCHP','SCHR','SCHV','SCHX','SCHZ','SCIF','SCPB','SCZ','SDIV','SDOG','SDY','SGOL','SH','SHM','SHV','SHY','SHYG','SIL','SJB','SJNK','SKYY','SLV','SLYG','SMH','SNLN','SOXX','SPHD','SPHQ','SPLV','SPY','SPYG','SRLN','STIP','STPZ','SUB','SVXY','SWH','TAN','TBF','TFI','THD','TILT','TIP','TLH','TLO','TLT','TLTD','TLTE','TOTL','TTFS','TUR','TUZ','UGA','UNG','USL','USMV','USO','UUP','VAW','VB','VBK','VBR','VCIT','VCLT','VCR','VCSH','VDC','VDE','VEA','VEU','VFH','VGIT','VGK','VGLT','VGSH','VGT','VHT','VIG','VIIX','VIS','VIXY','VLUE','VMBS','VNM','VNQ','VNQI','VO','VOE','VONG','VONV','VOO','VOOG','VOT','VOX','VPL','VPU','VRP','VSS','VT','VTEB','VTI','VTIP','VTV','VUG','VV','VWO','VWOB','VXF','VXUS','VXX','VXZ','VYM','WDTI','WOOD','XBI','XES','XHB','XIV','XLB','XLE','XLF','XLG','XLI','XLK','XLP','XLU','XLV','XLY','XME','XMLV','XOP','XPH','XRT','ZROZ'])


imp_csv_dbupdate(indir='E:/_data/stock/intraday/intraday_google_usetf2.h5',
                  outdir='E:/_data/stock/intraday/q5min/us/etf/', symbols=sym1  ,intype='hdfs')



pf.imp_hdfs_db_dumpinfo()



sym= np.array(['A','AA','AAL','AAP','AAPL','AAXJ','ABBV','ABC','ABT','ACAD','ACC','ACGL','ACH','ACHC','ACHN','ACM','ACN','ACOR','ACWI','ACWV','ACWX','ADAP','ADBE','ADI','ADM','ADP','ADRO','ADS','ADSK','ADT','AEE','AEG','AEM','AEP','AER','AES','AET','AFG','AFL','AFSI','AGG','AGIO','AGN','AGNC','AGQ','AGR','AGU','AGZ','AIG','AIMT','AIV','AIZ','AJG','AKAM','ALB','ALDR','ALGN','ALK','ALKS','ALL','ALLE','ALLY','ALNY','ALR','ALV','ALXN','AMAT','AMCX','AME','AMG','AMGN','AMJ','AMLP','AMP','AMRI','AMT','AMU','AMZN','AN','ANAC','ANIK','ANSS','ANTM','AOM','AON','AOR','AOS','APA','APC','APD','APH','AR','ARE','ARG','ARIA','ARMK','ARRS','ARW','ASH','ASHR','ASX','ATHN','ATMP','ATO','ATR','ATRA','ATVI','AU','AVAL','AVB','AVGO','AVT','AVY','AWK','AXON','AXP','AXS','AXTA','AYI','AZO','BA','BAB','BAC','BAP','BAX','BBBY','BBH','BBT','BBY','BCH','BCR','BDX','BEAV','BEN','BEP','BG','BGNE','BHI','BIB','BIIB','BIL','BIP','BIS','BIV','BK','BKFS','BKLN','BLK','BLL','BLUE','BLV','BMRN','BMS','BMY','BND','BNDS','BNDX','BNO','BOND','BPL','BPY','BR','BRFS','BRKR','BRO','BRX','BRZU','BSAC','BSCG','BSCH','BSCI','BSCJ','BSCK','BSCL','BSCM','BSJH','BSJI','BSJJ','BSMX','BSV','BSX','BUFF','BWA','BWX','BXLT','BXP','BZQ','C','CA','CAG','CAH','CAT','CB','CBG','CBM','CBOE','CBPO','CBS','CCE','CCI','CCJ','CCK','CCL','CDK','CDNS','CDW','CE','CEA','CELG','CEMP','CERN','CERS','CF','CFG','CG','CGW','CHAD','CHAU','CHD','CHK','CHRS','CHRW','CI','CIB','CINF','CIT','CIU','CL','CLB','CLLS','CLR','CLVS','CLX','CLY','CMA','CMCSA','CME','CMF','CMG','CMI','CMS','CNA','CNC','CNCO','CNHI','CNP','COF','COG','COH','COL','COMM','COO','COP','CORN','CORP','COST','COTY','CPB','CPG','CPGX','CPL','CPN','CPRT','CPT','CQH','CQP','CRI','CRM','CSCO','CSGP','CSJ','CSL','CSM','CSRA','CSX','CTAS','CTL','CTSH','CTXS','CUBE','CURE','CVC','CVE','CVS','CVX','CWB','CWI','CX','CXO','D','DAL','DBA','DBAW','DBB','DBC','DBEF','DBEM','DBEU','DBGR','DBJP','DBKO','DBO','DBP','DD','DDM','DDR','DE','DEG','DEM','DERM','DES','DEW','DFE','DFJ','DFS','DG','DGAZ','DGL','DGLD','DGRO','DGRW','DGX','DHI','DHR','DHS','DIA','DIG','DIS','DISCA','DISCK','DIV','DJP','DKS','DLN','DLPH','DLR','DLS','DLTR','DNB','DNO','DO','DOG','DON','DOV','DOW','DOX','DPS','DPZ','DRE','DRI','DRIP','DRN','DSI','DTE','DTN','DTO','DUG','DUK','DUST','DVA','DVAX','DVN','DVY','DVYE','DWM','DWTI','DWX','DXCM','DXD','DXJ','DZZ','EA','EBAY','EBS','ECA','ECH','ECL','ED','EDC','EDIT','EDU','EDV','EDZ','EELV','EEM','EEMA','EEMS','EEMV','EEP','EFA','EFAV','EFG','EFV','EFX','EFZ','EIDO','EIS','EIX','EL','ELS','EMB','EMC','EMLC','EMLP','EMN','EMR','ENDP','ENI','ENTA','EOG','EPC','EPHE','EPI','EPOL','EPP','EPU','EPZM','EQGP','EQIX','EQM','EQR','EQT','ERIE','ERJ','ERUS','ERX','ERY','ES','ESRX','ESS','ETE','ETFC','ETN','ETR','EUFN','EUM','EUO','EW','EWA','EWBC','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWN','EWP','EWQ','EWS','EWT','EWU','EWW','EWX','EWY','EWZ','EXC','EXEL','EXI','EXPD','EXPE','EXR','EZA','EZU','F','FANG','FAS','FAST','FAZ','FB','FBHS','FBR','FBT','FCAU','FCG','FCOM','FCX','FDC','FDL','FDN','FDS','FDT','FDX','FE','FENY','FEP','FEX','FEZ','FFIV','FGEN','FHLC','FIDU','FIS','FISV','FITB','FL','FLEX','FLIR','FLOT','FLR','FLS','FLT','FM','FMC','FNCL','FNDA','FNDF','FNDX','FNF','FNV','FNX','FOLD','FOX','FOXA','FPE','FPRX','FPX','FRC','FRT','FSLR','FSTA','FTA','FTC','FTI','FTNT','FTR','FTSD','FTSL','FTSM','FUTY','FV','FVD','FWP','FXA','FXB','FXC','FXD','FXE','FXG','FXH','FXI','FXL','FXO','FXP','FXU','FXY','FYX','G','GAS','GASL','GASX','GBT','GD','GDDY','GDX','GDXJ','GE','GG','GGG','GGOV','GGP','GIB','GIL','GILD','GIS','GLD','GLPG','GLW','GM','GMF','GMM','GNR','GNTX','GOLD','GOOG','GOOGL','GOVT','GPC','GPN','GPS','GRA','GREK','GRFS','GRMN','GS','GSG','GSLC','GSY','GT','GUNR','GUSH','GVI','GWW','GWX','GXC','GXP','H','HACK','HAL','HALO','HAR','HAS','HBAN','HBI','HCA','HCN','HCP','HD','HDG','HDGE','HDS','HDV','HEDJ','HEFA','HES','HEWG','HEWJ','HEZU','HFC','HIG','HII','HIW','HLF','HOG','HOLX','HON','HOT','HP','HPE','HPQ','HRB','HRL','HRS','HSIC','HST','HSY','HTZ','HUBB','HUM','HYD','HYEM','HYG','HYLS','HYMB','HYS','IAT','IAU','IBB','IBM','ICE','ICF','ICL','ICPT','IDLV','IDU','IDV','IDXX','IEF','IEFA','IEI','IEMG','IEO','IEP','IEUR','IEV','IEX','IEZ','IFF','IFGL','IGE','IGF','IGM','IGOV','IGV','IHDG','IHE','IHF','IHG','IHI','IHS','IJH','IJJ','IJK','IJR','IJS','IJT','ILB','ILF','ILMN','IM','IMGN','IMS','INCY','INDA','INDY','INGR','INO','INSM','INSY','INTC','INTU','IONS','IP','IPG','IPGP','IPS','IQDF','IR','IRM','ISRG','ISTB','IT','ITA','ITB','ITC','ITCI','ITE','ITIP','ITM','ITOT','ITR','ITW','IUSG','IUSV','IVE','IVV','IVW','IVZ','IWB','IWC','IWD','IWF','IWM','IWN','IWO','IWP','IWR','IWS','IWV','IWY','IXC','IXJ','IXP','IXUS','IYC','IYE','IYF','IYG','IYH','IYJ','IYK','IYM','IYR','IYT','IYW','IYY','IYZ','JAH','JAZZ','JBHT','JBLU','JCI','JDST','JEC','JHX','JJC','JKD','JKE','JKHY','JLL','JNJ','JNK','JNPR','JNUG','JO','JPM','JUNO','JWN','K','KAR','KB','KBE','KBWB','KERX','KEY','KEYS','KHC','KIE','KIM','KITE','KKR','KLAC','KMB','KMI','KMX','KO','KOF','KORS','KR','KRC','KRE','KSS','KSU','KT','KWEB','KXI','L','LABD','LABU','LAMR','LAZ','LB','LEA','LEG','LEMB','LEN','LGND','LH','LII','LKQ','LLL','LLTC','LLY','LM','LMCA','LMCK','LMT','LNC','LNG','LNT','LOW','LPL','LPT','LQD','LRCX','LTPZ','LUK','LULU','LUV','LVLT','LXRX','LYB','LYV','M','MA','MAA','MAC','MACK','MAN','MAR','MAS','MAT','MBB','MBLY','MBT','MCD','MCHI','MCHP','MCK','MCO','MCRB','MD','MDLZ','MDT','MDVN','MDY','MDYG','MDYV','MELI','MET','MGC','MGK','MGM','MGNX','MGV','MHG','MHK','MIC','MIDD','MIK','MINT','MJN','MKC','MKL','MKTX','MLM','MLPI','MLPN','MMC','MMM','MNK','MNKD','MNST','MO','MON','MOO','MOS','MPC','MPEL','MPLX','MRK','MRKT','MRO','MRVL','MS','MSCI','MSFT','MSI','MSM','MT','MTB','MTD','MTN','MTUM','MU','MUB','MUR','MVV','MXIM','MYL','N','NANR','NAVI','NBL','NCLH','NDAQ','NEAR','NEE','NEM','NEU','NFLX','NFX','NI','NIB','NK','NKE','NKTR','NLNK','NLSN','NLY','NNN','NOBL','NOC','NOV','NOW','NRG','NSC','NTAP','NTG','NTRS','NUAN','NUE','NUGT','NVAX','NVDA','NVR','NWL','NWSA','NYCB','O','OA','OAK','OC','ODFL','OEF','OGE','OHI','OI','OIH','OIL','OKE','OKS','OMC','OMER','ONCE','ONEQ','OPHT','OPK','ORCL','ORI','ORLY','OTEX','OUSA','OXY','PAA','PAC','PACB','PAGP','PALL','PANW','PAYX','PBA','PBCT','PBI','PBP','PBYI','PCAR','PCG','PCLN','PCY','PDCO','PDLI','PDP','PEG','PEJ','PEP','PEY','PF','PFE','PFF','PFG','PG','PGF','PGR','PGX','PH','PHB','PHDG','PHI','PHM','PHYS','PICK','PID','PIE','PII','PIN','PINC','PJP','PKG','PKI','PKW','PLD','PM','PNC','PNR','PNRA','PNW','PNY','POST','PPC','PPG','PPH','PPL','PPLT','PRF','PRGO','PRTA','PRU','PSA','PSL','PSLV','PSO','PSP','PSQ','PSX','PSXP','PTLA','PVH','PWB','PWR','PWV','PX','PXD','PXF','PXH','PYPL','PZA','Q','QAI','QCOM','QDEF','QDF','QGEN','QID','QIHU','QLD','QQEW','QQQ','QRVO','QSR','QUAL','QUNR','R','RACE','RAD','RAI','RARE','RCL','RDUS','RDY','RE','REG','REGL','REGN','REM','RF','RFG','RGA','RGEN','RGI','RHI','RHS','RHT','RIG','RIGS','RJF','RL','RLGY','RLY','RLYP','RMD','RNR','ROK','ROL','ROP','ROST','RPG','RPM','RPV','RRC','RS','RSG','RSP','RSX','RTN','RUSL','RUSS','RVNC','RWM','RWO','RWR','RWX','RYE','RYH','RYT','RYU','RZV','SABR','SAGE','SBAC','SBH','SBIO','SBNY','SBUX','SCG','SCHA','SCHB','SCHC','SCHD','SCHE','SCHF','SCHG','SCHH','SCHM','SCHO','SCHP','SCHR','SCHV','SCHW','SCHX','SCHZ','SCI','SCIF','SCO','SCPB','SCZ','SDIV','SDOG','SDOW','SDS','SDY','SE','SEE','SEF','SEIC','SEP','SERV','SGEN','SGOL','SH','SHI','SHLX','SHM','SHV','SHW','SHY','SHYG','SIG','SIL','SIVB','SIX','SJB','SJM','SJNK','SJR','SKF','SKX','SKYY','SLB','SLG','SLV','SLW','SLYG','SMG','SMH','SNA','SNDK','SNI','SNLN','SNPS','SO','SON','SOXL','SOXS','SOXX','SPB','SPG','SPGI','SPHD','SPHQ','SPIL','SPLK','SPLS','SPLV','SPR','SPXL','SPXS','SPXU','SPY','SPYG','SQM','SQQQ','SRC','SRCL','SRE','SRLN','SRPT','SRS','SRTY','SSNC','SSO','ST','STE','STI','STIP','STJ','STLD','STM','STPZ','STT','STWD','STX','STZ','SUB','SVXY','SWH','SWK','SWKS','SWN','SXL','SYF','SYK','SYMC','SYY','T','TAN','TAP','TARO','TBF','TBPH','TBT','TDC','TDG','TE','TEAM','TECH','TECL','TEL','TFI','TFSL','TFX','TGNA','TGT','THD','THS','TIF','TILT','TIP','TJX','TKC','TLH','TLO','TLT','TLTD','TLTE','TMF','TMK','TMO','TMV','TNA','TOL','TOTL','TQQQ','TRGP','TRIP','TRMB','TROW','TRQ','TRU','TRV','TS','TSCO','TSN','TSO','TSRO','TSS','TSU','TTC','TTFS','TUR','TUZ','TVIX','TWC','TWM','TWTR','TWX','TXN','TXT','TYC','TYL','TZA','UA','UA.C','UAL','UBIO','UCO','UDOW','UDR','UGA','UGAZ','UGI','UGLD','UGP','UHAL','UHS','ULTA','ULTI','UMC','UNG','UNH','UNM','UNP','UPRO','UPS','URBN','URE','URI','URTY','USB','USDU','USL','USLV','USMV','USO','UTHR','UTX','UUP','UVXY','UWM','UWTI','UYG','V','VAL','VAR','VAW','VB','VBK','VBR','VCIT','VCLT','VCR','VCSH','VDC','VDE','VEA','VER','VEU','VFC','VFH','VGIT','VGK','VGLT','VGSH','VGT','VHT','VIAB','VIG','VIIX','VIP','VIPS','VIS','VIXY','VLO','VLUE','VMBS','VMC','VNM','VNO','VNQ','VNQI','VNTV','VO','VOE','VONG','VONV','VOO','VOOG','VOOV','VOT','VOX','VOYA','VPL','VPU','VRP','VRSK','VRSN','VRTX','VSS','VT','VTEB','VTI','VTIP','VTR','VTV','VUG','VV','VWO','VWOB','VXF','VXUS','VXX','VXZ','VYM','VZ','WAB','WAT','WBA','WBC','WCN','WDAY','WDC','WDTI','WEC','WES','WF','WFC','WFM','WFT','WGP','WHR','WLK','WLTW','WM','WMB','WMT','WOOD','WOOF','WPC','WPZ','WR','WRB','WREI','WRI','WRK','WSM','WSO','WST','WTR','WU','WUBA','WWAV','WY','WYN','WYNN','XBI','XEC','XEL','XES','XHB','XIV','XL','XLB','XLE','XLF','XLG','XLI','XLK','XLNX','XLP','XLRN','XLU','XLV','XLY','XME','XMLV','XNCR','XOM','XON','XOP','XPH','XRAY','XRT','XRX','XT','XYL','Y','YANG','YHOO','YINN','YNDX','YPF','YUM','YZC','ZAYO','ZBH','ZBK','ZION','ZIOP','ZNH','ZROZ','ZTS'])





imp_csv_dbupdate(indir='E:/_data/stock/intraday/intraday_google_us.h5',
                  outdir='E:/_data/stock/intraday/q5min/us/stock/', symbols=sym  ,intype='hdfs')







def imp_db_hdfs_tonumpy()





Source de data

dest de data










df2['open']= 5

df=df2

from dateutil import parser


df.date= [util.datenumpy_todatetime(x)for x in  df.date.values ]



x= df.date.values[0]

datetime.fromtimestamp(x.astype('O')/1e9)


'''




'''
############## What to put in BColz/ zarr
   3D Array:  Time x Asset x Open Close High Low
   1 million quote per day
   2500 * 5= 12500







'''
##






type(df2.date.values[0])

df2

from dateutil.parser import parse

dateutil.parser.parse()

import pandas as pd


df3= pd.concat([df, df2], axis=0)


df4= df3.sort('date')



type(df.date.values[0])


from dateutil import parser
parser.parse("9/27/2016  10:55:00 PM")






   from_zone = tz.gettz(fromzone);   tozone = tz.gettz(tozone)
   dateparse= lambda x: (pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(tozone))
   # dateparse= lambda x: parse(x, tzinfos=from_zone).astimezone(to_zone)

   df = pd.read_csv(file1,sep=',',header=header, names=cols, parse_dates={'date': coldate }, date_parser=dateparse)
   df.date= [pd.to_datetime((str(x)[:-6])) for x in  df.date]
   df.date= [x.to_datetime() for x in  df.date]

   df.columns = [  x.lower() for x in df.columns.values ]
   # df.columns = ['date', 'symbol','open','high','low','close','volume']

   if util.find('symbol', df.columns.values) < 0 :
     df= util.pd_addcolumn(df, 'symbol')
     df['symbol']= sym

   return df


import util


df.drop(df.columns[[0]], axis=1, inplace=True)


'''
have long hungered for the ultimate, super-fast, super-scaleable data storage solution. I have used relational databases, kdb, flatfiles, and binary files. In the end, I used binary files in my research language of choice. My advice is to KISS. The choice of storage is actually not that critical (unless maybe you're working with options tick data). What is critical is how you decide to splay the data.

If you look at kdb, it can actually be quite slow if you don't splay (segment) the data for your particular need. It just gives you a fast management layer, but it is up to you to design the data storage on disk for your need. What you are trying to do is store the data in such a way so that you group together the data that you need and minimize the amount of extra data that has to be read off disk.

For me, I found storing data in binary format in the language that I do research in is the least amount of overhead. Managing a simple splay is easy. One key is don't be afraid to store multiple copies of your data for different research tasks, so long as the creation of the copies is driven off of one golden source. So for example, if you very often need all ticks for one stock for the past 5 years, then I would splay by stock. But if you also need all stocks for a given day, then I would store another dataset that splays by day. Process and store the data in a way that will be most useful to you.

If you are a big institution, then by all means spend the big $ to get kdb and hire a hotshot q programmer (b/c you are probably not going to figure it out on your own very easily). It is quite nice. But, if you are an individual, do the simple thing and move on to more interesting work.
'''


# create an in-memory numpy container
a = np.arange(10)
In [4]:
# create an in-memory carray container
b = bcolz.carray(a)

a= np.array( ['dsfsfksfksfufshjsfjjksfklsf sjfsdfs']*10*1000*1000 )


# create an in-memory carray container
b = bcolz.carray(a)




# create an on-disk carray container
c = bcolz.carray(a, rootdir='test3.bcolz', mode='w' )
c.flush()



d= bcolz.carray()


import util

df= util.pd_array_todataframe()


file1= 'E:/_data/stock/intraday/20161207_etf/NKE_20160928_300_6000.csv'
import bcolz
import pandas as pd

df = pd.read_csv(file1, delimiter=',')


ct = bcolz.ctable.fromdataframe(df, rootdir='dataframe.bcolz', mode='w' )











#---------------------- Ok,

Generally speaking, what was really important in this one was to find a way to cross validate(1st problem!)
and retain features (or interactions of them ) and then again there was the big difference
between the offers in the training and test set (2nd problem!).

For the first one we generally used a 1-vs-rest offers
' approach to test the AUC and sometimes even derivatives of that. For the second (problem) we tried to maximize the with-in offers' auc (how well the offers score individually irrespective of the rest) and the total AUC (e.g. how the different offers blend together) as separate objectives.

We used 3 (conceptually) different approaches (and some other minor blends):

1. Train with similar offers
2. Train with whether the customer would have bought the product anyway
3. Assume that some features work for all offers in the same way (like: if you bought the product before, that increases the probability of becoming/staying a repeater)

I can't stress enough how important it was to treat high-transaction IDs separate ' \
     'from "regular" IDs.
I shot up 150 places on the leader board just by doing that.

Then, breaking out training/testing by offer department was what made the rest of the difference.



--------------------------------------------------------------------------------------------------


Feature engineering

Feature engineering will be important in this competition, no matter the language or algorithms used.


We will generate the following features:

has_bought_company: the number of times a shopper has bought from the company on offer
has_bought_company_a: the total amount the shopper has bought from the company on offer
has_bought_company_q: the quantity of items bought from the company on offer.
has_bought_company_30: the number of times a shopper has bought from the company on offer in the 30 days before the date the coupon was offered.
has_bought_company_60: the number of times a shopper has bought from the company on offer in the 60 days before the date the coupon was offered.
...
has_bought_company_180: 180 days before
has_never_bought_company: a negative feature for when the shopper has never bought from the company on offer before.
These same features for:

has_bought_category: the number of times a shopper has bought from the category on offer
has_bought_brand: the number of times a shopper has bought from the brand on offer
Combinations of these:

has_bought_company_brand_category: if this feature is present the shopper has bought from the company, brand, and category on offer.
has_never_bought_company_brand: negative feature for the combination of brand and company purchase history.
Offer-related:

offer_value: The value of the coupon offer
offer_quantity: The number of products to redeem with the coupon
Total shopper spend:

total_shopper_spend: We take the total amount spend by the shopper in the reduced dataset.
Can you name some other possibly interesting features to generate?

Vowpal Wabbit

We train Vowpal Wabbit using quantile regression, 40 passes and a learning rate of 0.85. We turn the predictions into Kaggle's submission format.

Feature visualisation

Using the output from Vowpal Wabbit's wrapper vw-varinfo we generate the feature relevance plot below (code to generate this included):





#Kaggle Challenge:
#'Reduce the data and generate features' by Triskelion
#Very mediocre and hacky code, single-purpose, but pretty fast

from datetime import datetime, date
from collections import defaultdict

loc_offers = "kaggle_shop\\offers.csv"
loc_transactions = "kaggle_shop\\transactions.csv"
loc_reduced = "kaggle_shop\\reduced2.csv" # will be created


def data_csv_mapreduce(file_category, file_transact, file_reduced):
  ''' Reduce Data by filtering on some Category '''
  start = datetime.now()

  #Parse all categories and comps on offer in a dict
  offers_cat, offers_co = {}, {}
  for i, line in enumerate(open(file_category)):
    ll=  line.split(",")
    offers_cat[ ll[1] ] = 1
    offers_co[  ll[3] ] = 1


  #open output file
  with open(file_reduced, "wb") as outfile:
    #go through transactions file and reduce
    jj_new = 0
    for i, line in enumerate(open(file_transact)):
      if i == 0: outfile.write(line) #print header
      else:
        ll=  line.split(",")
        if ll[3] in offers_cat or ll[4] in offers_co:    #Condition  Filter : if category in offers dict
          outfile.write( line )
          jj_new += 1

      #progress
      if i % 5000000 == 0:  print i, jj_new, datetime.now() - start
  print i, jj_new, datetime.now() - start



def data_csv_mapreduce_chunk(csv_bigfile):
  chunksize =     10 * 10 ** 6
  df= pd.read_csv(csv_bigfile, chunksize=chunksize, lineterminator=',')




  df.to_csv(filenew, sep=',')



#reduce_data(loc_offers, loc_transactions, loc_reduced)

def diff_days(s1,s2):
	date_format = "%Y-%m-%d"
	a = datetime.strptime(s1, date_format)
	b = datetime.strptime(s2, date_format)
	delta = b - a
	return delta.days


loc_train = "kaggle_shop\\trainHistory.csv"
loc_test = "kaggle_shop\\testHistory.csv"
loc_transactions = "kaggle_shop\\reduced2.csv"
loc_out_train = "kaggle_shop\\train.vw"
loc_out_test = "kaggle_shop\\test.vw"
def generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test):
	#keep a dictionary with the offerdata
	offers = {}
	for e, line in enumerate( open(loc_offers) ):
		row = line.strip().split(",")
		offers[ row[0] ] = row

	#keep two dictionaries with the shopper id's from test and train
	train_ids = {}
	test_ids = {}
	for e, line in enumerate( open(loc_train) ):
		if e > 0:
			row = line.strip().split(",")
			train_ids[row[0]] = row
	for e, line in enumerate( open(loc_test) ):
		if e > 0:
			row = line.strip().split(",")
			test_ids[row[0]] = row
	#open two output files
	with open(loc_out_train, "wb") as out_train, open(loc_out_test, "wb") as out_test:
		#iterate through reduced dataset
		last_id = 0
		features = defaultdict(float)
		for e, line in enumerate( open(loc_transactions) ):
			if e > 0: #skip header
				#poor man's csv reader
				row = line.strip().split(",")
				#write away the features when we get to a new shopper id
				if last_id != row[0] and e != 1:

					#generate negative features
					if "has_bought_company" not in features:
						features['never_bought_company'] = 1

					if "has_bought_category" not in features:
						features['never_bought_category'] = 1

					if "has_bought_brand" not in features:
						features['never_bought_brand'] = 1

					if "has_bought_brand" in features and "has_bought_category" in features and "has_bought_company" in features:
						features['has_bought_brand_company_category'] = 1

					if "has_bought_brand" in features and "has_bought_category" in features:
						features['has_bought_brand_category'] = 1

					if "has_bought_brand" in features and "has_bought_company" in features:
						features['has_bought_brand_company'] = 1

					outline = ""
					test = False
					for k, v in features.items():

						if k == "label" and v == 0.5:
							#test
							outline = "1 '" + last_id + " |f" + outline
							test = True
						elif k == "label":
							outline = str(v) + " '" + last_id + " |f" + outline
						else:
							outline += " " + k+":"+str(v)
					outline += "\n"
					if test:
						out_test.write( outline )
					else:
						out_train.write( outline )
					#print "Writing features or storing them in an array"
					#reset features
					features = defaultdict(float)
				#generate features from transaction record
				#check if we have a test sample or train sample
				if row[0] in train_ids or row[0] in test_ids:
					#generate label and history
					if row[0] in train_ids:
						history = train_ids[row[0]]
						if train_ids[row[0]][5] == "t":
							features['label'] = 1
						else:
							features['label'] = 0
					else:
						history = test_ids[row[0]]
						features['label'] = 0.5

					#print "label", label
					#print "trainhistory", train_ids[row[0]]
					#print "transaction", row
					#print "offers", offers[ train_ids[row[0]][2] ]
					#print

					features['offer_value'] = offers[ history[2] ][4]
					features['offer_quantity'] = offers[ history[2] ][2]
					offervalue = offers[ history[2] ][4]

					features['total_spend'] += float( row[10] )

					if offers[ history[2] ][3] == row[4]:
						features['has_bought_company'] += 1.0
						features['has_bought_company_q'] += float( row[9] )
						features['has_bought_company_a'] += float( row[10] )

						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_company_30'] += 1.0
							features['has_bought_company_q_30'] += float( row[9] )
							features['has_bought_company_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_company_60'] += 1.0
							features['has_bought_company_q_60'] += float( row[9] )
							features['has_bought_company_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_company_90'] += 1.0
							features['has_bought_company_q_90'] += float( row[9] )
							features['has_bought_company_a_90'] += float( row[10] )
						if date_diff_days < 180:
							features['has_bought_company_180'] += 1.0
							features['has_bought_company_q_180'] += float( row[9] )
							features['has_bought_company_a_180'] += float( row[10] )

					if offers[ history[2] ][1] == row[3]:

						features['has_bought_category'] += 1.0
						features['has_bought_category_q'] += float( row[9] )
						features['has_bought_category_a'] += float( row[10] )
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_category_30'] += 1.0
							features['has_bought_category_q_30'] += float( row[9] )
							features['has_bought_category_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_category_60'] += 1.0
							features['has_bought_category_q_60'] += float( row[9] )
							features['has_bought_category_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_category_90'] += 1.0
							features['has_bought_category_q_90'] += float( row[9] )
							features['has_bought_category_a_90'] += float( row[10] )
						if date_diff_days < 180:
							features['has_bought_category_180'] += 1.0
							features['has_bought_category_q_180'] += float( row[9] )
							features['has_bought_category_a_180'] += float( row[10] )
					if offers[ history[2] ][5] == row[5]:
						features['has_bought_brand'] += 1.0
						features['has_bought_brand_q'] += float( row[9] )
						features['has_bought_brand_a'] += float( row[10] )
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_brand_30'] += 1.0
							features['has_bought_brand_q_30'] += float( row[9] )
							features['has_bought_brand_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_brand_60'] += 1.0
							features['has_bought_brand_q_60'] += float( row[9] )
							features['has_bought_brand_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_brand_90'] += 1.0
							features['has_bought_brand_q_90'] += float( row[9] )
							features['has_bought_brand_a_90'] += float( row[10] )
						if date_diff_days < 180:
							features['has_bought_brand_180'] += 1.0
							features['has_bought_brand_q_180'] += float( row[9] )
							features['has_bought_brand_a_180'] += float( row[10] )
				last_id = row[0]
				if e % 100000 == 0:
					print e
#generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test)
loc_preds = "kaggle_shop\\shop.preds.txt"
loc_test = "kaggle_shop\\testHistory.csv"
loc_submission = "kaggle_shop\\kaggle.submission2.csv"

def generate_submission(loc_preds, loc_test, loc_submission):
	preds = {}
	for e, line in enumerate( open(loc_preds) ):
		row = line.strip().split(" ")
		preds[ row[1] ] = row[0]


	with open(loc_submission, "wb") as outfile:
		for e, line in enumerate( open(loc_test) ):
			if e == 0:
				outfile.write( "id,repeatProbability\n" )
			else:
				row = line.strip().split(",")
				if row[0] not in preds:
					outfile.write(row[0]+",0\n")
				else:
					outfile.write(row[0]+","+preds[row[0]]+"\n")
#generate_submission(loc_preds, loc_test, loc_submission)






















In Python to reduce from 20GB to about 1GB (349.655.789 lines to 15.349.956 lines) or "the category subset" as BreakfastPirate calls it:

from datetime import datetime

loc_offers = "kaggle_shop\\offers.csv"
loc_transactions = "kaggle_shop\\transactions.csv"
loc_reduced = "kaggle_shop\\reduced2.csv" # will be created

def reduce_data(loc_offers, loc_transactions, loc_reduced):

  start = datetime.now()
  #get all categories on offer in a dict
  offers = {}
  for e, line in enumerate( open(loc_offers) ):
    offers[ line.split(",")[1] ] = 1
  #open output file
  with open(loc_reduced, "wb") as outfile:
    #go through transactions file and reduce
    reduced = 0
    for e, line in enumerate( open(loc_transactions) ):
      if e == 0:
        outfile.write( line ) #print header
      else:
        #only write when category in offers dict
          if line.split(",")[3] in offers:
            outfile.write( line )
            reduced += 1
      #progress
      if e % 5000000 == 0:
        print e, reduced, datetime.now() - start
  print e, reduced, datetime.now() - start

reduce_data(loc_offers, loc_transactions, loc_reduced)

if you want to reduce the data with company, change:

offers[ line.split(",")[1] ] = 1

to:

offers[ line.split(",")[3] ] = 1

and:

if line.split(",")[3] in offers

to:

if line.split(",")[4] in offers















