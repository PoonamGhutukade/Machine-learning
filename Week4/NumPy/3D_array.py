
import re
from Week4.Utility.Util import UtilClass

"""18. Write a Python program to create a 3-D array with ones on a diagonal and zeros elsewhere. 
Expected Output:
[[ 1. 0. 0.]
[ 0. 1. 0.] 
[ 0. 0. 1.]]"""


class ThreeDArray:

    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class functions from utility
    def calling(self):
        print("\nCreate array ")
        input1 = input("\nEnter the array start value:")
        input2 = input("Enter the end value:")
        array_created = self.obj1.matrix_creation(input1,input2)
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\nGive proper dimension for matrix:")
            num1 = input("Enter the 1st dimension:")
            num2 = input("Enter the 2nd dimension:")
            result = self.obj1.reshape_matrix(array_created, num1,num2)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Original Array: \n", result)
                # it will take as as header comment to input matrix
                res = self.obj1.identitymatrix(result)
                print("3-D array with ones on a diagonal: \n", res)


# class Object created to call its methods
obj = ThreeDArray()
obj.calling()