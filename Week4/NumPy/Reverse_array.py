"""
4. Write a Python program to reverse an array (first element becomes last).
Original array:
[12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]
Reverse array:
[37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12]
"""
from Week4.Utility.Util import UtilClass

import re


class NumpyClass1:
    # class constructor
    def __init__(self):
        self.obj1 = UtilClass()

    def calling(self):
        print("\nPut values from 12 to 37 to reverse array ")
        # It display number from 12 to 37
        array_created = self.obj1.matrix_creation()
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            print("\nOriginal Matrix:", array_created)
            # call reverse method
            print("Reverse array:", self.obj1.matrix_reverse(array_created))


# class object created to call its methods
obj = NumpyClass1()
obj.calling()
