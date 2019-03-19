"""5. Write a Python program to create a 2d array with 1 on the border and 0 inside.
Expected Output:
Original array:
[[ 1. 1. 1. 1. 1.]
[ 1. 1. 1. 1. 1.]
[ 1. 1. 1. 1. 1.]
[ 1. 1. 1. 1. 1.]
[ 1. 1. 1. 1. 1.]]
1 on the border and 0 inside in the array
[[ 1. 1. 1. 1. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 0. 0. 0. 1.]
[ 1. 1. 1. 1. 1.]]

"""
from Week4.Utility.Util import UtilClass
import re


class Matrix:
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    def calling(self):
        print("\nPut values from 1 to 25  ")
        # It display number from 1 to 25
        array_created = self.obj1.matrix_creation()
        str1 = str(array_created)
        # check output correct or not
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\n 5 * 5 Dimension matrix")
            matrix_of_one = self.obj1.matrix_one_creation(array_created)
            result = self.obj1.reshape_matrix(matrix_of_one)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape given matrix into 5 * 5 or given format: \n", result)
                print("\n 1 on the border and 0 inside in the array")
                # Give result as zero inside and border with one
                # fill matrix [1(from 1st row at top):-1(from bottom 1 row), 1(1st column):-1(from right side )] = 0
                result[1:-1, 1:-1] = 0
                print(result)


# class object created to call its methods
obj = Matrix()
obj.calling()
