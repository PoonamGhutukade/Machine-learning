""" 2. Create a 3x3 matrix with values ranging from 2 to 10.
Expected Output:
[[ 2 3 4]
[ 5 6 7]
[ 8 9 10]]
"""
from Week4.Utility.Util import UtilClass
import re


class NumpyClass1:
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class function from utility
    def calling(self):
        print("\nPut values for reshape matrix from 2 to 10")
        array_created = self.obj1.matrix_creation()
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            print("\nNew Matrix:\n", array_created)
            print("\n3 * 3 Dimension matrix")
            result = self.obj1.reshape_matrix(array_created)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape given matrix into 3*3 format: \n", result)


# class object created to call its methods
obj = NumpyClass1()
obj.calling()
