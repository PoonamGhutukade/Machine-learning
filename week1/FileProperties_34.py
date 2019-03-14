""" 34. Write a Python program to retrieve file properties. """

import os.path
import time

print("\nFile: ", __file__)

print("\nAccessed time: ", time.ctime(os.path.getatime(__file__)))
print("\nModification time: ", time.ctime(os.path.getmtime(__file__)))
print("\nChange time: ", time.ctime(os.path.getctime(__file__)))
print("\nsize: ", os.path.getsize(__file__))

print("\n Path:", os.path.join(os.path.dirname(__file__)))
