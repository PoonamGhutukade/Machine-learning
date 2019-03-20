import numpy as np
import re
from Week4.Utility.Util import UtilClass

"""17. Write a Python program to change the data type of an array. 
Expected Output:
[[ 2 4 6]
[ 6 8 10]] 
Data type of the array x is: int32 
New Type: float64 
[[ 2. 4. 6.] 
[ 6. 8. 10.]]"""


class StoreArray:

    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class functions from utility
    def calling(self):
        print("\ncreate matrix to store it into file: ")
        array_created = self.obj1.matrix_creation()
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            print("\nNew Matrix:\n", array_created)
            print("\nGive proper dimension for matrix:")
            result = self.obj1.reshape_matrix(array_created)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Original Matrix: \n", result)
                # it will take as as header comment to input matrix
                res = self.obj1.change_dt(result)
                print("Array with other DataType: \n", res)


# class Object created to call its methods
obj = StoreArray()
obj.calling()
