""" 39. Write a Python program to find files and skip directories of a given directory.
"""
import os
# only show files in given directory i.e skips the directories
print([file for file in os.listdir('/home/admin1/Coader') if os.path.isfile(os.path.join('/home/admin1/Coader', file))])
