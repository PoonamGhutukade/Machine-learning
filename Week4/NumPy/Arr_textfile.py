"""15. Write a Python program to save a NumPy array to a text file. """
import numpy as np
import re
from Week4.Utility.Util import UtilClass


class StoreArray:
    # arr1 = np.array([10,20,30,40][20,40,50,60])
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class function from utility
    def calling(self):
        print("\nCreate array to store it into file: ")
        input1 = input("\nEnter the start value for array:")
        input2 = input("Enter the end value:")
        array_created = self.obj1.matrix_creation(input1, input2)

        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\nGive proper dimension for matrix:")
            num1 = input("Enter the 1st dimension:")
            num2 = input("Enter the 2nd dimension:")
            result = self.obj1.reshape_matrix(array_created, num1, num2)

            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape array into given format: \n", result)
                # it will take as as header comment to input matrix
                header = 'c1 c2 c3 '
                # file only create once and then it  will update with new inputs
                # savetxt function is used to store data into file
                np.savetxt('file12.txt', result, fmt=" %d ", header=header)
    # class object created to call its methods


# class Object created to call its methods
obj = StoreArray()
obj.calling()
