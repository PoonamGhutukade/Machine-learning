""" 16. Write a Python program to get the current username
"""
import getpass
import os
print("Current User Name: ",getpass.getuser())

print("---------------------OR----------------------------")
# It shows current user name
print(os.environ['USER'])