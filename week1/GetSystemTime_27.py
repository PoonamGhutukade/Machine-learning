""" 27. Write a Python program to get the system time. """

import time
import datetime

print()
# The system time is important for debugging, network information, random number seeds,
# or something as simple as program performance
print(time.ctime())


print()
print(datetime.datetime.now())

