"""11. Write a Python program to find the number of elements of an array, length of one array element in bytes and
 total bytes consumed by the elements.
Expected Output:
Size of the array: 3
Length of one array element in bytes: 8
Total bytes consumed by the elements of the array: 24


12. Write a Python program to find common values between two arrays.
Expected Output:
Array1: [ 0 10 20 40 60]
Array2: [10, 30, 40]
Common values between two arrays:
[10 40]

13. Write a Python program to find the set difference of two arrays. The set difference will return the sorted,
unique values in array1 that are not in array2.
Expected Output:
Array1: [ 0 10 20 40 60 80]
Array2: [10, 30, 40, 50, 70, 90]
Set difference between two arrays:
[ 0 20 60 80]

14. Write a Python program to find the set exclusive-or of two arrays. Set exclusive-or will return the sorted,
unique values that are in only one (not both) of the input arrays.
Array1: [ 0 10 20 40 60 80]
Array2: [10, 30, 40, 50, 70]
Unique values that are in only one (not both) of the input arrays:
[ 0 20 30 50 60 70 80]

15. Write a Python program compare two arrays using numpy.
Array a: [1 2]
Array b: [4 5]
a > b
[False False]
a >= b
[False False]
a < b
[ True True]
a <= b
[ True True]
"""
import numpy as np
import re
from Week4.Utility.Util import UtilClass


class SizeofElements:
    # class constructor
    def __init__(self):
        # create obj of util class
        self.obj1 = UtilClass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Find the number of elements,length in byte and total Bytes of an array""\n"
                      "2. Operation on array :""\n"
                      "3. Compare two arrays using numpy""\n"
                      "4. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        # find the number of elements of an array
                        self.obj1.array_size_ele()

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        """find common values between two arrays
                        find the set difference of two arrays
                        Set exclusive-or will return the sorted, unique values """
                        self.obj1.operation_on_two_arrays()

                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        # compare two arrays using numpy
                        self.obj1.compare_two_array()

                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        # Exit from code
                        exit()

                        print("_______________________________________________________________________________")

                    else:
                        print("Plz enter valid choice: ")

                    acc = str(input("IF you want to continue: type yes "))
                    # if acc == 'y' and acc == 'yes':
                    if re.match(acc, 'y'):
                        continue
                    elif re.match(acc, 'yes'):
                        continue
                    elif re.match(acc, 'n'):
                        break
                    elif re.match(acc, 'no'):
                        break
                    else:
                        print("Give proper input")
                        continue

                else:
                    raise ValueError

            except ValueError as e:
                print("\nInvalid Input", e)


# create obj of our class to call its methods
obj = SizeofElements()
obj.calling()
