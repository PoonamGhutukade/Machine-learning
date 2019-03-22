"""
3. Write a Python program to create a null vector of size 10 and update sixth value to 11.
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
        input1 = input("\nEnter the matrix start value:")
        input2 = input("Enter the matrix end value:")
        array_created = self.obj1.matrix_creation(input1, input2)
        str1 = str(array_created)
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            # create null vector
            result = self.obj1.null_vector_creation(array_created)

            print("\nOriginal null vector array :", result)
            # update null vector sixth value to 11
            print("Update array: ", self.obj1.update_matrix(result))


# class object created to call its methods
obj = NumpyClass1()
obj.calling()
