""" Write a Python program to calculate number of days between two dates.
Sample dates : (2014, 7, 2), (2014, 7, 11)
Expected output : 9 days """

import datetime
from datetime import date

"""
    @param date1 and date2
    calculate diff between date1 and date2
"""


def diff_days(date1, date2):
    a = date1
    b = date2
    return (a - b).days


print()
print(diff_days((date(2016, 10, 12)), date(2015, 12, 10)))
print(diff_days((date(2015, 3, 23)), date(2016, 3, 23)))
