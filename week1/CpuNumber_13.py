"""  13. Write a Python program to find out the number of CPUs using. """

# for python2

import multiprocessing
import os

# cpu_count() show the numbers of cpu
print("CPU Count: ", multiprocessing.cpu_count())

print("-----------------------OR---------------------------------")

# for python3



print("CPU Count:", os.cpu_count())
