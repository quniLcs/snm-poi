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


if __name__ == '__main__':
    date = str2date('Tue Apr 03 18:17:18 +0000 2012')
    print(date)
