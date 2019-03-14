""" 22. Write a Python program to get the command-line arguments
(name of the script, the number of arguments, arguments) passed to a script.
"""

import sys

# his is the path of the script with script name:
print("\n Script name:", sys.argv[0])

"""followint both lines doing the same work
print(__file__)
print(main.__file__)
"""
print("\n Number of arguments:", len(sys.argv))
print("\n Argument List:", str(sys.argv))

# take two input values from command line
print("Enter values for addition: ")
x = int(input(sys.argv[1]))
y = int(input(sys.argv[2]))
z = x + y
print("\n Addition of two variables: ",z)
