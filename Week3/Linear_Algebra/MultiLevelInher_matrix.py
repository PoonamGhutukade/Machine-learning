from Week3.Utility.Util import UtilClass
import re


# Parent class
class Parent:
    # global declaration
    matrix1 = [[1, 7, 3],
               [4, 5, 6],
               [7, 8, 9]]

    matrix2 = [[5, 8, 1],
               [4, 5, 6],
               [7, 8, 9]]

    vector1 = [1, 2, 3]

    result = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]


# child class i.e second class
class Child(Parent):
    # class constructor
    def __init__(self):
        # create obj of utility class
        self.obj = UtilClass()

        # display method to display original matrix and vector
    def display(self):
            print("\nOriginal Matrix1: ")
            for temp11 in self.matrix1:
                print(temp11)
            # print("__________________________________________")
            print("Original Matrix2: ")
            for temp11 in self.matrix2:
                print(temp11)
            # print("__________________________________________")
            print("Original Vector: ")
            print(self.vector1)
            # print("__________________________________________")

        # Vector and matrix multiplication
    def mat_vect_multi(self):
            # vector multi
            retvalue = self.obj.vectormultiplication(self.vector1, self.matrix1)
            print("\nVector and matrix1 multiplication:")
            for temp11 in retvalue:
                print(temp11)
            # print("__________________________________________")


# last class
class Class3(Child):
    # class constructor
    def __init__(self):
        super().__init__()

    # matrix1 and matrix2 multiplication
    def matmultiplication(self):
        result12 = self.obj.multiplication1(self.matrix1, self.matrix2, self.result)
        print("\nMatrix1 and matrix2 multiplication:")
        for temp in result12:
            print(temp)


# once obj created  of class3 it automatically call init method, also it call all methods of its parent class also
obj1 = Class3()

while True:
    try:
        print()
        print("1. Display original matrix and vector "
              "\n""2. Vector and matrix multiplication ""\n"
              "3. matrix1 and matrix2 multiplication""\n""4. Exit")
        ch = input("Enter choice:")
        choice = int(ch)
        if ch.isdigit():
            if choice == 1:
                # 2nd class method called by 3rd class
                obj1.display()
                print("_______________________________________________________________________________")

            elif choice == 2:
                obj1.mat_vect_multi()
                print("_______________________________________________________________________________")

            elif choice == 3:
                # last class method
                obj1.matmultiplication()
                print("_______________________________________________________________________________")

            elif choice == 4:
                exit()

            else:
                print("Plz enter valid choice: ")

            acc = str(input("IF you want to continue: type yes otherwise no "))
            # if acc == 'y' and acc == 'yes':
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
        print("\nInvalid Input: ", e)

# # 2nd class method called by 3rd class
# obj1.display()
# obj1.mat_vect_multi()
#
# # last class method
# obj1.matmultiplication()
