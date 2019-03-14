"""  Write a Python program to print the calendar of a given month and year.
Note : Use 'calendar' module.
"""

import calendar

# Both values are in integer
year = int(input("\nInput the year: "))
month = int(input("Input the month: "))
print()
print(calendar.month(year, month))