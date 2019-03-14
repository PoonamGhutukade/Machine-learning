""" 11. Write a Python program to check whether a file exists. """

import os.path

"""
    main() function check file is exist or not
"""


def main():
    # Take input from user
    fname = input("Enter the file name: ")
    ftype = input("Enter the file extention: ")
    newfile = fname + "." + ftype
    # check is file exist? if it give true means exist otherwise false (Not exist)
    print(os.path.isfile(newfile))


main()

print()
file1 = open("f1.txt", "r")
print("In file: f1.txt ")
# Read content in file
print("One Line: ", file1.readlines())
print()

# Use One function at a time
"""print("Read file: ",file1.read())
  OR
print("All Line",file1.readlines())
print()"""
