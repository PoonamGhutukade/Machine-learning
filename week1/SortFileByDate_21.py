""" 21. Write a Python program to sort files by date. """

import glob
import os

file = glob.glob("*.txt")
# getmtime ->  is a  'get modification time'. ( Sort by modification time)
file.sort(key=os.path.getmtime)
print("\n".join(file))
