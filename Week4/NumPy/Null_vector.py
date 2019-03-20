"""
3. Write a Python program to create a null vector of size 10 and update sixth value to 11.
[ 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Update sixth value to 11
[ 0. 0. 0. 0. 0. 0. 11. 0. 0. 0.]
"""
from Week4.Utility.Util import UtilClass
import re


class NumpyClass1:
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    def calling(self):

        print("\nPut values for null vector from 0 to 10")
        array_created = self.obj1.matrix_creation()
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            #print("\nNew Matrix:\n", array_created)
            # create null vector
            result = self.obj1.null_vector_creation(array_created)

            print("\nOriginal null vector array :",result)
            # update null vector sixth value to 11
            print("Update array: ", self.obj1.update_matrix(result))


# class object created to call its methods
obj = NumpyClass1()
obj.calling()