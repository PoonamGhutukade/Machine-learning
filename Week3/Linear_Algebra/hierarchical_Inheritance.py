from Week3.Utility.Util import UtilClass
import re


# Parent class has 2 child i.e. hierarchical inheritance implementation
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


# child class derived variables from parent class
class Child(Parent):
    # class constructor
    def __init__(self):
        # create obj of utility class
        self.obj = UtilClass()

    # display method to display original matrix and vector
    def display(self):
        print("Original Matrix1: ")
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
    def mat_addition(self):
        # Matrix addition
        value1 = self.obj.matrix_addtn(self.matrix1, self.matrix2, self.result)

        # following for loop show addition of two matrices
        for temp4 in value1:
            print(temp4)
        # print("__________________________________________")


# 2nd child class derived variables from parent class
class Child2(Parent):
    # class constructor
    def __init__(self):
        self.obj = UtilClass()

    # matrix1 and matrix2 multiplication
    def matmultiplication(self):
        result12 = self.obj.multiplication1(self.matrix1, self.matrix2, self.result)
        print("\nMatrix1 and matrix2 multiplication:")
        for temp in result12:
            print(temp)


# once the object is created  of child class, it automatically call it's init method
# 1st child class object
obj1 = Child()
# 2nd child class obj
obj2 = Child2()

while True:
    try:
        print()
        print("1. Display original matrix and vector "
              "\n""2. Matrix Addition ""\n"
              "3. matrix1 and matrix2 multiplication""\n""4. Exit")
        ch = input("Enter choice:")
        choice = int(ch)
        if ch.isdigit():
            if choice == 1:
                obj1.display()
                print("_______________________________________________________________________________")

            elif choice == 2:
                obj1.mat_addition()
                print("_______________________________________________________________________________")

            elif choice == 3:
                obj2.matmultiplication()
                print("_______________________________________________________________________________")

            elif choice == 4:
                exit()

            else:
                print("Plz enter valid choice: ")

            acc = str(input("IF you want to continue: type yes "))
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
        print("\nInvalid Input", e)
