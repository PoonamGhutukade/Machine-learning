"""16. Write a Python program to create a contiguous flattened array.
Original array:
[[10 20 30]
[20 40 50]]
New flattened array:
[10 20 30 20 40 50]
 """
import numpy as np
import re
from Week4.Utility.Util import UtilClass


class ArrayOperation:
    # arr1 = np.array([10,20,30,40][20,40,50,60])
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class functions from utility
    def calling(self):
        print("\nCreate an array: ")
        input1 = input("\nEnter the start value for array:")
        input2 = input("Enter the end value:")
        array_created = self.obj1.matrix_creation(input1, input2)
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\nGive proper dimension:")
            num1 = input("Enter the 1st dimension:")
            num2 = input("Enter the 2nd dimension:")
            result = self.obj1.reshape_matrix(array_created, num1, num2)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape array into given format: \n", result)

                # it will take as as header comment to input matrix
                res = self.obj1.flattendarr(result)
                print("Flatted array: ",res)


# class Object created to call its methods
obj = ArrayOperation()
obj.calling()
