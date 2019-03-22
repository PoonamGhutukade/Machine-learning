"""
4. Write a Python program to reverse an array (first element becomes last).

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
        input1 = input("\nEnter the matrix start value:")
        input2 = input("Enter the matrix end value:")
        array_created = self.obj1.matrix_creation(input1, input2)
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
