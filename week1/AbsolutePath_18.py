""" 18. Write a Python program to get an absolute file path. """
import os

"""
    @param file
    to get absolute path of that file 
"""


def newf(file):
    # it return detailed path
    return os.path.abspath(file)


file = input("File name: ")
z = os.path.exists(file)
try:
    f = open(file, 'rb')
    print("File exist", z)
    print("\nAbsolute path for", file, "is:", newf(file))
    f.close()
except FileNotFoundError:
    print("\nInvalid file Name")
