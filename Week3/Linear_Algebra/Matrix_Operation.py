"""Matrix operation"""
from Week3.Utility.Util import UtilClass
import re


class LinearAlgebra:

    # class Constructor
    def __init__(self, matrix1, matrix2, result, vector):
        # self.newname = Parameter
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.result = result
        self.vector = vector
        # create obj of utility class
        self.obj = UtilClass()

    # all operation on matrix
    def matrixoperation(self):
        while True:
            try:
                print()
                print("1. Add matrices ""\n"
                      "2. Perform scalar multiplication of matrix and a number"
                      "\n""3. Perform multiplication of given matrix and vector""\n"
                      "4. Multiply matrices in problem 1""\n"
                      "5. Find inverse matrix of matrix ""\n"
                      "6. Find transpose matrix""\n""7. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        # 1st addition
                        print("addition of two matrix:\n")
                        # Matrix addition
                        value1 = self.obj.matrix_addtn(self.matrix1, self.matrix2, self.result)

                        # following for loop show addition of two matrices
                        for temp4 in value1:
                            print(temp4)

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        # # 2nd  scalar Matrix multiplication
                        print("Scalar multiplication of two matrix:\n")
                        value3 = self.obj.matrix_scalar_multi(self.matrix1, self.result)
                        # It show addition of two matrices
                        # use loop to display matrix in proper format
                        for temp5 in value3:
                            print(temp5)
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        print("Vector multiplication of two matrix:\n")
                        value5 = self.obj.vectormultiplication(self.vector, self.matrix1)
                        # use loop to display matrix in proper format
                        for temp5 in value5:
                            print(temp5)
                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        # 4. multiplication
                        print("Multiplication of two matrix:\n")
                        value2 = self.obj.matrix_multiplication(self.matrix1, self.matrix2)
                        # It show matrix_multiplication
                        # use loop to display matrix in proper format
                        for temp3 in value2:
                            print(temp3)
                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        print("Inverse of matrix:\n")
                        value1 = self.obj.inverse_matrix(self.matrix1)
                        for temp in value1:
                            print(temp)

                    elif choice == 6:

                        print("Transpose of matrix:")
                        self.obj.transpose_matrix(self.matrix1)

                        print("---------------------------OR-----------------------")

                        print("Transpose of matrix:")
                        value1 = self.obj.transpose(self.matrix1)
                        # use loop to display matrix in proper format
                        for temp in value1:
                            print(temp)

                        print("---------------------------OR-----------------------")

                        print("\nTranspose of matrix:")
                        self.obj.trans_matrix(self.matrix1)
                    elif choice == 7:
                        exit()
                        print("_______________________________________________________________________________")

                    else:
                        print("Plz enter valid choice: ")

                    acc = str(input("IF you want to continue: type yes "))
                    if re.match(acc, 'y'):
                        continue
                    elif re.match(acc, 'yes'):
                        continue
                    elif re.match(acc, 'n'):
                        break
                    elif re.match(acc, 'no'):
                        break
                    else:
                        print("Give proper input")
                        continue

                else:
                    raise ValueError
            except ValueError as e:
                print("\nInvalid Input", e)


matrix1 = [[1, 7, 3],
           [4, 5, 6],
           [7, 8, 9]]

matrix2 = [[5, 8, 1],
           [4, 5, 6],
           [7, 8, 9]]

result = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]

vector = [1, 2, 3]

# class object created and pass variable values to init constructor
obj = LinearAlgebra(matrix1, matrix2, result, vector)

# call matrix multiplication method
obj.matrixoperation()
