
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:29:50 2017
"""


import arrow
from datetime import datetime
import numpy as np
import scipy as sci
import pandas as pd





def datestring_todatetime(x, fmt="%Y%m%d%H%M%S" ) :
    pass







def datetime_tointhour(datelist1):
    if isinstance(datelist1, datetime.date):
        return int(datelist1.strftime("%Y%m%d%H%M%S"))
    yy2 = []
    for x in spdateref1:
        yy2.append(
            x.year * 10000 * 10000 * 100
            + x.month * 10000 * 10000
            + x.day * 100 * 10000
            + x.hour * 10000
            + x.minute * 100
            + x.second
        )
    return np.array(yy2)






############################################################################################
fm t ='YYYY-MM-DD'
def date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH:mm:SS') :
    d d= arrow.get(str_t1, fmt) - arrow.get(str_t0, fmt)
    return dd.total_seconds()


def date_diffstart(t) : return date_diffsecond(str_t1=t, str_t0=t0)
def date_diffend(t) :   return date_diffsecond(str_t1=t1, str_t0=t)


def np_dict_tolist(dd) :
    return [ val  for _, val in list(dd.items()) ]

def np_dict_tostr_val(dd) :
    return ','.join([ str(val)  for _, val in list(dd.items()) ])

def np_dict_tostr_key(dd) :
    return ','.join([ str(key)  for key ,_ in list(dd.items()) ])


###################Faster one   ############################################################
# 'YYYY-MM-DD    HH:mm:ss'
# "0123456789_10_11
import arrow, copy
"""
def day(s):    return int(s[8:10])
def month(s):  return int(s[5:7])
def year(s):   return int(s[0:4])
def hour(s):   return int(s[11:13])
"""


cache_weekda y= {}
def weekday(s, fmt='YYYY-MM-DD', i0=0, i1=10):
    ###Super Fast because of caching
    s 2= s[i0:i1]
    try :
        return  cache_weekday[s2]
    except KeyError:
        w d= arrow.get(s2, fmt).weekday()
        cache_weekday[s2 ]= wd
    return wd



def season(d): m=  int( d[5:7])
    if m > 3 and m  < 10:
        return 1
    else:
        return 0


def daytime(d):
    h= int(d[11:13])
    if   h < 11 :
        return 0
    elif h < 14 :
        return 1  # lunch
    elif h < 18 :
        return 2  # Afternoon
    elif h < 21 :
        return 3  # Dinner
    else :
        return 4  # Night


def pd_date_splitall(df, coldate='purchased_at') :
    df= copy.deepcopy(df)
    df['year']= df[coldate].apply(year)
    df['month']= df[coldate].apply(month)
    df['day']= df[coldate].apply(day)
    df['weekday']= df[coldate].apply(weekday)
    df['daytime']= df[coldate].apply(daytime)
    df['season']= df[coldate].apply(season)
    return df


def date_tosecond(dd,  format1="YYYYMMDDHHmm", unit="second") :
 try :
   t= arrow.get(str(dd), "YYYYMMDDHHmmss")
   return t.timestamp
 except :
    try :
       t= arrow.get(str(dd), "YYYYMMDDHHmm")
       return t.timestamp
    except :
       return -1

df['datesec']= df['datetime'].apply(  date_tosecond  )






#########################Date manipulation ##########################################
def date_earningquater(t1):
    qdate = quater = None
    if (t1.month == 10 and t1.day >= 14) or (t1.month == 1 and t1.day < 14) or t1.month in [11, 12]:
        if t1.month in [10, 11, 12]:
            qdate = datetime.datetime(t1.year + 1, 1, 14)
        else:
            qdate = datetime.datetime(t1.year, 1, 14)
        quater = 4

    if (t1.month == 1 and t1.day >= 14) or (t1.month == 4 and t1.day < 14) or t1.month in [2, 3]:
        qdate = datetime.datetime(t1.year, 4, 14)
        quater = 1

    if (t1.month == 4 and t1.day >= 14) or (t1.month == 7 and t1.day < 14) or t1.month in [5, 6]:
        qdate = datetime.datetime(t1.year, 7, 14)
        quater = 2

    if (t1.month == 7 and t1.day >= 14) or (t1.month == 10 and t1.day < 14) or t1.month in [8, 9]:
        qdate = datetime.datetime(t1.year, 10, 14)
        quater = 3

    nbday = (qdate - t1).days
    return quater, nbday, qdate


def date_is_3rdfriday(s):
    d = datetime.strptime(s, "%b %d, %Y")
    return d.weekday() == 4 and 14 < d.day < 22


def date_option_expiry(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    if date.day <= day:
        nbday = day - date.day
        datexp = datetime.datetime(date.year, date.month, day)
    else:
        if date.month == 12:
            day = 21 - (calendar.weekday(date.year + 1, 1, 1) + 2) % 7
            datexp = datetime.datetime(date.year + 1, 1, day)
        else:
            day = 21 - (calendar.weekday(date.year, date.month + 1, 1) + 2) % 7
            datexp = datetime.datetime(date.year, date.month + 1, day)

        nbday = (datexp - date).days

    return nbday, datexp


def date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate):
    return util.np_find(datetime.date(intradaydate[kintraday]), dailydate)


def date_find_kintraday_fromdate(d1, intradaydate1, h1=9, m1=30):

    d1 = datetime.datetime(d1.year, d1.month, d1.day, h1, m1)
    return util.np_find(d1, intradaydate1)


def date_find_intradateid(datetimelist, stringdate=None):
    if stringdate is None:
        stringdate = ["20160420223000"]
    for t in stringdate:
        tt = datestring_todatetime(t)
        k = util.np_find(tt, datetimelist)
        print(str(k) + ",", tt)


# kday= date_find_kday_fromintradaydate(k, spdateref2, spdailyq.date)


def datetime_convertzone1_tozone2(tt, fromzone="Japan", tozone="US/Eastern"):
    import pytz, dateutil

    tz = pytz.timezone(fromzone)
    tmz = pytz.timezone(tozone)

    if type(tt) == datetime:
        localtime = tz.localize(tt).astimezone(tmz)
        return dateutil.parser.parse(localtime.strftime("%Y-%m-%d %H:%M:%S"))

    t2 = []
    for t in tt:
        localtime = tz.localize(t).astimezone(tmz)
        t2.append(dateutil.parser.parse(localtime.strftime("%Y-%m-%d %H:%M:%S")))
    return t2

    # --------- Find Daily Open / Close Time  --------------------------------------


def date_extract_dailyopenclosetime(spdateref1, market="us"):
    if market == "us":
        topenh = 9
        topenm = 30
        tcloseh = 16
        tclosem = 00
        spdailyopendate = []
        for k, t in enumerate(spdateref1):
            if t.hour == topenh and t.minute == topenm:
                spdailyopendate.append(k)
        spdailyopendate = np.array(spdailyopendate)

        spdailyclosedate = []
        for k, t in enumerate(spdateref1):
            if t.hour == tcloseh and t.minute == tclosem:
                spdailyclosedate.append(k)
        spdailyclosedate = np.array(spdailyclosedate)

        return spdailyopendate, spdailyclosedate
    else:
        return None, None


def datetime_tostring(tt):
    if isinstance(tt, datetime.date):
        return tt.strftime("%Y%m%d")
    if isinstance(tt, np.datetime64):
        t = pd.to_datetime(str(tt))
        return t.strftime("%Y%m%d")

    if isinstance(tt[0], np.datetime64):
        tt = pd.to_datetime(str(tt))
    date2 = []
    for t in tt:
        date2.append(t.strftime("%Y%m%d"))
    return date2


def datestring_todatetime(datelist1, format1="%Y%m%d"):
    if isinstance(datelist1, str):
        return parser.parse(datelist1)
    date2 = []
    for s in datelist1:
        date2.append(parser.parse(s))
        # date2.append(datetime.datetime.strptime(s, format1))
    return date2


def datetime_todate(tt):
    if isinstance(tt, datetime.datetime):
        return datetime.date(tt.year, tt.month, tt.day)
    if isinstance(tt, np.datetime64):
        tt = pd.to_datetime(str(tt))
        return datetime.date(tt.year, tt.month, tt.day)

    if isinstance(tt[0], np.datetime64):
        tt = pd.to_datetime(tt)
    date2 = []
    for t in tt:
        date2.append(datetime.date(t.year, t.month, t.day))
    return date2


def datetime_toint(datelist1):
    if isinstance(datelist1, datetime.date):
        return int(datelist1.strftime("%Y%m%d"))
    date2 = []
    for t in datelist1:
        date2.append(int(t.strftime("%Y%m%d")))
    return date2


def datetime_tointhour(datelist1):
    if isinstance(datelist1, datetime.date):
        return int(datelist1.strftime("%Y%m%d%H%M%S"))
    yy2 = []
    for x in spdateref1:
        yy2.append(
            x.year * 10000 * 10000 * 100
            + x.month * 10000 * 10000
            + x.day * 100 * 10000
            + x.hour * 10000
            + x.minute * 100
            + x.second
        )
    return np.array(yy2)


# noinspection PyTypeChecker
def dateint_tostring(datelist1, format1="%b-%y"):
    if isinstance(datelist1, int):
        return dateint_todatetime(datelist1).strftime(format1)
    date2 = []
    for s in datelist1:
        date2.append(dateint_todatetime(s).strftime(format1))
    return np.array(date2)


"""
>>> import datetime
>>> datetime.datetime.strptime('20-Nov-2002','%d-%b-%Y').strftime('%Y%m%d')
'20021120'
Formats -

%d - 2 digit date

%b - 3-letter month abbreviation

%Y - 4 digit year

%m - 2 digit month
"""


def dateint_todatetime(datelist1):
    if isint(datelist1):
        return parser.parse(str(datelist1))
    date2 = []
    for s in datelist1:
        date2.append(parser.parse(str(s)))
        # date2.append(datetime.datetime.strptime(s, format1))
    return date2


def datenumpy_todatetime(tt, islocaltime=True):
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    if type(tt) == np.datetime64:
        if islocaltime:
            return datetime.datetime.fromtimestamp(tt.astype("O") / 1e9)
        else:
            return datetime.datetime.utcfromtimestamp(tt.astype("O") / 1e9)
    elif type(tt[0]) == np.datetime64:
        if islocaltime:
            v = [datetime.datetime.fromtimestamp(t.astype("O") / 1e9) for t in tt]
        else:
            v = [datetime.datetime.utcfromtimestamp(t.astype("O") / 1e9) for t in tt]
        return v
    else:
        return tt  # datetime case


def datetime_tonumpypdate(t, islocaltime=True):
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    return np.datetime64(t)


def date_diffindays(intdate1, intdate2):
    dt = dateint_todatetime(intdate2) - dateint_todatetime(intdate1)
    return dt.days


def date_finddateid(date1, dateref):
    i = util.np_findfirst(date1, dateref)
    if i == -1:
        i = util.np_findfirst(date1 + 1, dateref)
    if i == -1:
        i = util.np_findfirst(date1 - 1, dateref)
    if i == -1:
        i = util.np_findfirst(date1 + 2, dateref)
    if i == -1:
        i = util.np_findfirst(date1 - 2, dateref)
    if i == -1:
        i = util.np_findfirst(date1 + 3, dateref)
    if i == -1:
        i = util.np_findfirst(date1 - 3, dateref)
    if i == -1:
        i = util.np_findfirst(date1 + 5, dateref)
    if i == -1:
        i = util.np_findfirst(date1 - 5, dateref)
    if i == -1:
        i = util.np_findfirst(date1 + 7, dateref)
    if i == -1:
        i = util.np_findfirst(date1 - 7, dateref)
    return i


def datestring_toint(datelist1):
    if isinstance(datelist1, str):
        return int(datelist1)
    date2 = []
    for s in datelist1:
        date2.append(int(s))
    return date2


def date_as_float(dt):
    if isint(dt):
        dt = dateint_todatetime(dt)
    size_of_day = 1.0 / 366.0
    size_of_second = size_of_day / (24.0 * 60.0 * 60.0)
    days_from_jan1 = dt - datetime.datetime(dt.year, 1, 1)
    if not isleap(dt.year) and days_from_jan1.days >= 31 + 28:
        days_from_jan1 += timedelta(1)
    return dt.year + days_from_jan1.days * size_of_day + days_from_jan1.seconds * size_of_second


# start_date = datetime(2010,4,28,12,33)
# end_date = datetime(2010,5,5,23,14)
# difference_in_years = date_as_float(end_time) - date_as_float(start_time)


def date_todatetime(tlist):
    return [datetime.datetime(t.year, t.month, t.day) for t in tlist]


def date_removetimezone(datelist):
    return [pd.to_datetime((str(x)[:-6])).to_datetime for x in datelist]


def datediff_inyear(startdate, enddate):
    return date_as_float(startdate) - date_as_float(enddate)


def date_generatedatetime(start="20100101", nbday=10, end=""):
    from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR

    start = datestring_todatetime(start)
    if end == "":
        end = date_add_bdays(start, nbday - 1)  #  + datetime.timedelta(days=nbday)
    date_list = list(rrule(DAILY, dtstart=start, until=end, byweekday=(MO, TU, WE, TH, FR)))

    return np.array(date_list)


def date_add_bdays(from_date, add_days):
    if type(from_date) == int:
        current_date = dateint_todatetime(from_date)
    elif type(from_date) == str:
        current_date = datestring_todatetime(from_date)
    else:
        current_date = from_date

    business_days_to_add = add_days
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5:
            continue  # sunday = 6
        business_days_to_add -= 1
    return current_date


def date_getspecificdate(
    datelist,
    datetype1="yearend",
    outputype1="intdate",
    includelastdate=True,
    includefirstdate=False,
):
    vec2 = []

    if isint(datelist[0]):
        datelist = dateint_todatetime(datelist)

    t0 = datelist[0]
    if datetype1 == "monthend":
        for i, t in enumerate(datelist):
            #      print(datetime_tostring([t0, t]))
            if t.month != t0.month:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if datetype1 == "2monthend":
        for i, t in enumerate(datelist):
            if t.month != t0.month and np.mod(t0.month, 2) == 0:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if datetype1 == "3monthend":
        for i, t in enumerate(datelist):
            if t.month != t0.month and np.mod(t0.month, 3) == 0:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if datetype1 == "4monthend":
        for i, t in enumerate(datelist):
            if t.month != t0.month and np.mod(t0.month, 4) == 0:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if datetype1 == "6monthend":
        for i, t in enumerate(datelist):
            if t.month != t0.month and np.mod(t0.month, 6) == 0:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if datetype1 == "monthstart":
        for i, t in enumerate(datelist):
            if t.month != t0.month:
                vec2.append([i, t])
                # month has change
            t0 = t

    if datetype1 == "yearstart":
        vec2.append([0, t0])
        for i, t in enumerate(datelist):
            if t.year != t0.year:
                vec2.append([i, t])
                # month has change
            t0 = t

    if datetype1 == "yearend":
        for i, t in enumerate(datelist):
            if t.year != t0.year:
                vec2.append([i - 1, t0])
                # month has change
            t0 = t

    if includelastdate:
        vec2.append([len(datelist) - 3, datelist[-1]])

    if outputype1 == "intdate":
        vec2 = np.array(vec2)
        vec2 = np.array(vec2[:, 0], dtype="int")
        return vec2
    else:
        return np.array(vec2)


def date_alignfromdateref(array1, dateref):  # 2 column array time, data
    # --------Align the array1= date/raw  with same date than dateref---------------------
    masset, _ = np.shape(array1)
    tmax = len(dateref)
    close = np.zeros((masset, tmax), dtype="float32")

    for k in range(1, masset):
        #  df= array1[k]
        for t in range(0, tmax):
            ix = np.argwhere(array1[0, :] == np.float(dateref[t]))  # Find the date index
            if len(ix) > 0:  # If found
                ix1 = ix[0, 0]
                close[k, t] = array1[k, ix1]  # Update the close value
                close[0, t] = array1[0, ix1]  # Update the date

    # Use Previous values to fill Zero value
    for k in range(1, masset):
        for t in range(0, tmax):
            if close[k, t] == 0:
                close[k, t] = close[k, t - 1]

    return close


@jit(numba.float32[:](numba.int32[:], numba.int32[:], int32, numba.float32[:]))
def _date_align(dateref, datei, tmax, closei):
    close2 = np.zeros(tmax, dtype=np.float16)
    for t in range(0, tmax):
        ix = np.argwhere(datei == dateref[t])  # Find the date index
        if len(ix) > 0:  # If found
            ix0 = ix[0]
            close2[t] = closei[ix0]  # Update the close value

    for t in range(0, tmax):
        if close2[t] == 0:
            close2[t] = close2[t - 1]
    return close2


# @jit
def date_align(quotes, dateref=None, datestart=19550101, type1="close"):
    """ #Aligne the price with the same dates date	year	month	day	d	open	close	high	low	volume	aclose """
    df0 = quotes[0]
    if dateref is None:
        dateref = np.array(df0.date.values)

    isnotint1 = not isint(dateref[0])
    if isnotint1:
        dateref = np.array(datetime_toint(dateref))
    if datestart != 19550101:
        dateref = dateref[dateref > datestart]

    masset = len(quotes)
    tmax = len(dateref)
    close = np.zeros((masset, tmax), dtype="float16")

    print(("Period: ", dateref[0], dateref[-1], len(dateref)))
    for k in range(0, masset):
        df = quotes[k]
        priceid = util.find(type1, df.columns)  # close price column
        closei = df.iloc[:, priceid].values  #!!!!! Otherwise 2  Columns Time Series
        if not isint(df.date.values[0]):
            datei = np.array(datetime_toint(df.date.values))
        else:
            datei = np.array(df.date.values)
        close[k, :] = _date_align(dateref, datei, tmax, closei)
    return np.array(close, dtype=np.float32), dateref

