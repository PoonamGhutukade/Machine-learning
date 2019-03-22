"""
6. Write a Python program to add a border (filled with 0's) around an existing array."""
from Week4.Utility.Util import UtilClass
import re
import numpy as np


class Matrix:
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    def calling(self):
        print("\nPut values from 1 to 9  ")
        # It display number from 1 to 9
        input1 = input("\nEnter the matrix start value:")
        input2 = input("Enter the matrix end value:")
        array_created = self.obj1.matrix_creation(input1, input2)
        str1 = str(array_created)
        # check output correct or not
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\n 3 * 3 Dimension matrix")
            matrix_of_one = self.obj1.matrix_one_creation(array_created)
            num1 = input("Enter the 1st dimension:")
            num2 = input("Enter the 2nd dimension:")
            result = self.obj1.reshape_matrix(matrix_of_one, num1, num2)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape given matrix into 3*3 or given format: \n", result)
                print("\n0 on the border and 1 inside in the array")
                # Give result as zero outside and border with zeroes
                # syntax->numpy.pad(array, pad_width, mode, **kwargs)
                result1 = np.pad(result, pad_width=1, mode='constant', constant_values=0)
                print(result1)


# class object created to call its methods
obj = Matrix()
obj.calling()
