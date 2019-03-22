"""
7. Write a Python program to create a 8x8 matrix and fill it with a checkerboard pattern.
Checkerboard pattern:"""
from Week4.Utility.Util import UtilClass
import re


class Checkerboard:
    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    def calling(self):
        print("\nPut values from 1 to 64  ")
        # It display number from 1 to 64
        input1 = input("\nEnter the matrix start value:")
        input2 = input("Enter the matrix end value:")
        array_created = self.obj1.matrix_creation(input1, input2)
        str1 = str(array_created)
        # check output correct or not
        if re.match(str1, 'None'):
            print("Output will not display")
        else:
            # print("\nNew Matrix:\n", array_created)
            print("\n 8 * 8 Dimension matrix")
            # whole matrix fill with zeroes
            matrix_of_one = self.obj1.null_vector_creation(array_created)
            num1 = input("Enter the 1st dimension:")
            num2 = input("Enter the 2nd dimension:")
            result = self.obj1.reshape_matrix(matrix_of_one, num1, num2)
            str2 = str(result)

            if re.match(str2, 'None'):
                print("Output will not display")
            else:
                print("Reshape given matrix into 8*8 or given format: \n", result)
                """ x[1::2, ::2] = 1 : Slice from 1st index row till 1+2+2… (repeated for 2nd iteration)and 
                                        fill all columns with 1 starting from 0th to 0+2+2… and so on.
                    
                    x[::2, 1::2] = 1 : Slice from 0th row till 0+2+2… and 
                                        fill all columns with 1 starting from 1 to 1+2+2+….. """
                result[::2, 1::2] = 1
                result[1::2, ::2] = 1
                print("Checkerboard pattern:\n", result)


# class object created to call its methods
obj = Checkerboard()
obj.calling()
