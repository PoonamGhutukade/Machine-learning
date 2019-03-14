""" 33. Write a Python program to get the users environment. """

import os
import textwrap
print()
print(os.environ)

print("-------------------------------------OR------------------------------")

user_env = ','.join(os.environ)

print(textwrap.fill(user_env, 90))