# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705,R0913
# -*- coding: utf-8 -*-
"""
import datetime
datetime.datetime.strptime('20-Nov-2002','%d-%b-%Y').strftime('%Y%m%d')
'20021120'
Formats -

%d - 2 digit date
%b - 3-letter month abbreviation
%Y - 4 digit year
%m - 2 digit month
%a

df = DataFrame(dict(date = date_range('20130101',periods=10)))
https://python-utils.readthedocs.io/en/latest/usage.html#quickstart
https://dateutil.readthedocs.io/en/stable/examples.html



"""

from datetime import datetime

import dateutil
import numpy as np
import pandas as pd


def pd_datestring_split(
        dfref, coldate, fmt="%Y-%m-%d %H:%M:%S", return_val="split"):
    """
      Parsing date
      'Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'
    :param datelist:
    :param fmt:
    :return:
    """
    fmt = None if fmt in [None, "auto"] else fmt
    if not isinstance(coldate, str):
        raise Exception("codlate must be string")
    df = pd.DataFrame(dfref[coldate])

    coldt = coldate + "_dt"
    df[coldt] = pd.to_datetime(
        df[coldate], errors="coerce", format=None, infer_datetime_format=True, cache=True
    )

    df[coldate + "_year"] = df[coldt].apply(lambda x: x.year)
    df[coldate + "_month"] = df[coldt].apply(lambda x: x.month)
    df[coldate + "_day"] = df[coldt].apply(lambda x: x.day)

    if return_val == "split":
        return df
    else:
        return df[[coldate + "_year", coldate + "_month", coldate + "_day"]]


def datestring_todatetime(datelist, fmt="%Y-%m-%d %H:%M:%S"):
    """
      Parsing date
      'Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'
    :param datelist:
    :param fmt:
    :return:
    """
    datenew = []
    if fmt == "auto":
        if isinstance(datelist, list):
            for x in datelist:
                try:
                    datenew.append(dateutil.parser.parse(x))
                except Exception as e:
                    datenew.append(pd.NaT)

            return datenew
        else:
            return dateutil.parser.parse(datelist)
    else:
        if isinstance(datelist, list):
            return [datetime.strptime(x, fmt) for x in datelist]
        else:
            return datetime.strptime(datelist, fmt)


def datetime_tostring(datelist, fmt="%Y-%m-%d %H:%M:%S"):
    """
  https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
  :param x:
  :param fmt:
  :return:
  """
    if isinstance(datelist, list):
        ll = [datetime.strftime(x, fmt) for x in datelist]
        return ll
    else:
        return datetime.strftime(datelist, fmt)


def datetime_tointhour(datelist):
    if not isinstance(datelist, list):
        x = datelist
        y = (
                x.year * 10000 * 10000 * 100
                + x.month * 10000 * 10000
                + x.day * 100 * 10000
                + x.hour * 10000
                + x.minute * 100
                + x.second
        )
        return y
    yy2 = []
    for x in datelist:
        yy2.append(
            x.year * 10000 * 10000 * 100
            + x.month * 10000 * 10000
            + x.day * 100 * 10000
            + x.hour * 10000
            + x.minute * 100
            + x.second
        )
    return np.array(yy2)


def datetime_toint(datelist):
    if not isinstance(datelist, list):
        x = datelist
        return x.year * 10000 + x.month * 100 + x.day
    yy2 = []
    for x in datelist:
        yy2.append(x.year * 10000 + x.month * 100 + x.day)
    return np.array(yy2)


def datetime_to_milisec(datelist):
    if not isinstance(datelist, list):
        return (datelist - datetime(1970, 1, 1)).total_seconds()
    else:
        ll = [(t - datetime.datetime(1970, 1, 1)).total_seconds()
              for t in datelist]
        return ll


def datetime_weekday(datelist):
    if not isinstance(datelist, list):
        return int(datelist.strftime("%w"))
    else:
        return [int(x.strftime("%w")) for x in datelist]


dd_weekday_cache = {}


def datetime_weekday_fast(dateval):
    """
      date values
    :param dateval:
    :return:
    """
    try:
        return dd_weekday_cache[dateval]
    except BaseException:
        d = datetime_weekday(dateval)
        dd_weekday_cache[dateval] = d
        return d


def datetime_quarter(datetimex):
    m = datetimex.month
    return int(m // 3) + 1


def dateime_daytime(datetimex):
    h = datetimex.hour
    if h < 11:
        return 0
    elif h < 14:
        return 1  # lunch
    elif h < 18:
        return 2  # Afternoon
    elif h < 21:
        return 3  # Dinner
    else:
        return 4  # Night


def datenumpy_todatetime(tt, islocaltime=True):
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    if isinstance(tt, np.datetime64):
        if islocaltime:
            return datetime.fromtimestamp(tt.astype("O") / 1e9)
        else:
            return datetime.utcfromtimestamp(tt.astype("O") / 1e9)
    elif isinstance(tt[0], np.datetime64):
        if islocaltime:
            v = [datetime.fromtimestamp(t.astype("O") / 1e9) for t in tt]
        else:
            v = [datetime.utcfromtimestamp(t.astype("O") / 1e9) for t in tt]
        return v
    else:
        return tt  # datetime case


def datetime_tonumpydate(t, islocaltime=True):
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    return np.datetime64(t)




def np_dict_tolist(dd):
    return [val for _, val in list(dd.items())]


def np_dict_tostr_val(dd):
    return ",".join([str(val) for _, val in list(dd.items())])


def np_dict_tostr_key(dd):
    return ",".join([str(key) for key, _ in list(dd.items())])
