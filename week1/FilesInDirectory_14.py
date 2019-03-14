""" 14. Write a Python program to list all files in a directory in Python. """

from os import listdir
from os.path import isfile, join
import os

# only show files in given directory
flist = [file for file in listdir('/home/admin1/Coader') if isfile(join('/home/admin1/Coader', file))]
print(flist)
print()

# It shoes all files and floders
print(os.listdir('/home/admin1/Coader'))
