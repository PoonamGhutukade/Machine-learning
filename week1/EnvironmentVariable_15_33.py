"""  15. Write a python program to access environment variables.
33. Write a Python program to get the users environment.
"""
import os

print("-------------------------------------------------")

# Access all envi variable
print("All evn Variables: ", os.environ)
print("-------------------------------------------------")

# Access particular env variable
print("Home: ", os.environ['HOME'])
print("-------------------------------------------------")
print("Path: ", os.environ['PATH'])
print("-------------------------------------------------")

# Current user name
print("User: ", os.environ['USER'])
