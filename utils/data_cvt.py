import numpy as np
from datetime import datetime


def str2date(utcTimestamp):
    
    month_str_to_month_num = {
        "Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12
    }
    
    time_data = utcTimestamp.split(' ')
    year, month, day, time = time_data[-1], time_data[1], time_data[2], time_data[3]
    hour, minute, second = time.split(':')
    
    return datetime(int(year), month_str_to_month_num[month], int(day), int(hour), int(minute), int(second))

def str2date_Bk(utcTimestamp):
    # 2010-10-17T01:48:53Z
    
    month_str_to_month_num = {
        "Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12
    }
    
    year, month, day, hour, minute, second =\
        utcTimestamp[:4], utcTimestamp[5:7], utcTimestamp[8:10], utcTimestamp[11:13], utcTimestamp[14:16], utcTimestamp[17:19]
    
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))