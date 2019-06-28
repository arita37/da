"""
Methods for reading the data from csv, xls, DB's etc, and writting to csv 
and does some preprocessing
"""

######## Read file and extract data pattern:  .csv,  .txt, .xls  ##################################
############## Excel processing #######################################################################


def xl_setstyle(file1):
    pass


def xl_val(ws, colj, rowi):
    pass

def xl_get_rowcol(ws, i0, j0, imax, jmax):
    pass


def xl_getschema(dirxl="", filepattern="*.xlsx", dirlevel=1, outfile=".xlsx"):
    pass

############## csv processing #######################################################################

def csv_dtypes_getdict(df=None, csvfile=None):
    pass

def csv_fast_processing():
    """
   http://word.bitly.com/post/74069870671/optimizing-text-processing

import sys
from collections import defaultdict
OUT_FILES = defaultdict(dict)

open_outfiles()  # open all files I could possibly need

for line in sys.stdin:
    # 1. parse line for account_id and metric_type
    key = line.split(',')
    account_id = key[ACCOUNT_ID_INDEX][1:] # strip leading quote

    # 2. write to appropriate file for account_id and metric_type
    OUT_FILES[account_id][key[METRIC_TYPE_INDEX]].write(line)

   close_outfiles()  # close all the files we opened

   """


def csv_col_schema_toexcel(
    dircsv="",
    filepattern="*.csv",
    outfile=".xlsx",
    returntable=1,
    maxrow=5000000,
    maxcol_pertable=90,
    maxstrlen="U80",
):
    """Take All csv in a folder and provide Table, Column Schema, type
 str(df[col].dtype)  USE str always, otherwise BIG Issue

METHOD FOR Unicode / ASCII issue
1. Decode early:  Decode to <type 'unicode'> ASAP
    df['PREF_NAME']=       df['PREF_NAME'].apply(to_unicode)
2. Unicode everywhere
3. Encode late :f = open('/tmp/ivan_out.txt','w')
                f.write(ivan_uni.encode('utf-8'))
 """
    pass



def csv_col_get_dict_categoryfreq(
    dircsv, filepattern="*.csv", category_cols=[], maxline=-1, fileencoding="utf-8"
):
    pass


def csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header=True, maxline=-1):
    """ Reduce Data Row by filtering on some Category
    file_category=  in1+ "offers.csv"
    ncol= 8
    catval_tokeep=[ {} for i in xrange(0, ncol)]
    for i, line in enumerate(open(file_category)):
      ll=  line.split(",")
      catval_tokeep[3][  ll[1] ]  = 1  # Offer_file_col1 --> Transact_file_col_4
      catval_tokeep[4][  ll[3] ] =  1  # Offer_file_col3 --> Transact_file_col_4

  def condfilter(colk, catval_tokeep) :
    if colk[3] in catval_tokeep[3] or colk[4] in catval_tokeep[4]: return True
    else: return False
  """
    pass
    """
  does not work, issue with character encoding....
      with open(fromfile, 'r') as f :
     with csv.reader(f,  delimiter=',' ) as reader :
      for ll in reader:
  """


def csv_analysis():
    pass
    """
   https://csvkit.readthedocs.io/en/540/tutorial/1_getting_started.html

   sudo pip install csvkit

   :return:
   """


def csv_row_reduce_line_manual(file_category, file_transact, file_reduced):
    """ Reduce Data by filtering on some Category """
    pass


def csv_row_mapreduce(dircsv="", outfile="", type_mapreduce="sum", nrow=1000000, chunk=5000000):
    pass
    """Take All csv in a folder and provide Table, Column Schema"""


def csv_pivotable(
    dircsv="",
    filepattern="*.csv",
    fileh5=".h5",
    leftX="col0",
    topY="col2",
    centerZ="coli",
    mapreduce="sum",
    chunksize=500000,
    tablename="df",
):
    """ return df Pivot Table from series of csv file (transfer to d5 temporary)

Edit: you can groupby/sum from the store iteratively since this "map-reduces" over the chunks:

reduce(lambda x, y: x.add(y, fill_value=0),
       (df.groupby().sum() for df in store.select('df', chunksize=50000)))

 """
    pass


def csv_bigcompute():
    pass

######################## DB related items #######################################################################
def db_getdata():
    pass


def db_sql():
    pass


def db_meta_add(
    metadb, dbname, new_table=("", []), schema=None, df_table_uri=None, df_table_columns=None
):
    """ Create Meta database to store infos on the tables : csv, zip, HFS, Postgres
ALL_DB['japancoupon']= {}
ALL_DB['japancoupon']['schema']=    df_schema
ALL_DB['japancoupon']['df_table_uri']= df_schema_dictionnary
ALL_DB['japancoupon']['df_table_columns']= df_schema_dict
        DBname, db_schema, db_table_uri, db_table_columns(dict_table->colum_list),
   """
    pass


def db_meta_find(ALLDB, query="", filter_db=[], filter_table=[], filter_column=[]):
    """ Find string in all the meta table name, column
  db_meta_find(ALLDB, query='bottler', filter_db=['cokeon'],   filter_table=['table'], filter_column=['table'] )
  dbname: should be exact name
  fitler_table: partial match is ok
  fitler_column : partial name is ok
  return   (dbname, meta_table_name,  meta_table_filtered_by_row_containing query)
  """
    pass


######################  Pre Processing  ###############################################################

def str_to_unicode(x, encoding="utf-8"):
    pass

def isnull(x):
    pass

"""
Methods for for feature extraction ???
"""
