
import numpy as np
import re
from Week4.Utility.Util import UtilClass


class ThreeDArray:

    # class constructor
    def __init__(self):
        # utility class objected created here
        self.obj1 = UtilClass()

    # call this class functions from utility # 19 to 21
    def calling(self):
        while True:
            try:
                print()
                print("1. Create an array as per requirement""\n"
                      "2. Concatenate 2-dimensional arrays""\n"
                      "3. Make an array immutable (read-only)""\n"
                      "4. Create an array of (3, 4) shape, multiply every element value by 3""\n"
                      "5. Convert a NumPy array into Python list structure ""\n"
                      "6. Add an extra column to an numpy array""\n"
                      "7. Remove specific elements in a numpy array""\n"
                      "8. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        """19. Write a Python program to create an array which looks like below array. """
                        array = np.tri(4, 3, -1)
                        print("Final array: \n", array)
                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        # 20
                        data1 = np.array([[0, 1, 3], [5, 7, 9]])
                        data2 = np.array([[0, 2, 4], [6, 8, 10]])
                        print("original arrays:\n",data1,"\n",data2)
                        print("Concreate array:",self.obj1.concreate_data(data1, data2))
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        # 21
                        print("\nPut values from 0 to 10")
                        input1 = input("\nEnter the start value of array:")
                        input2 = input("Enter the end value:")
                        array_created = self.obj1.matrix_creation(input1, input2)
                        str1 = str(array_created)
                        if re.match(str1, 'None'):
                            print("Output will not display")
                        else:
                            # print("\nNew Matrix:\n", array_created)
                            # create null vector
                            result = self.obj1.null_vector_creation(array_created)

                            print("\nOriginal null vector array :", result)
                            # print("try to change value for null")
                            a = np.zeros((3, 3))
                            # here we set writable = false , so we can only read our data
                            result.flags.writeable = False
                            result[0] = 1
                            print(result)

                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        # 22
                        print("\nPut values from 1 to 12  ")

                        input1 = input("\nEnter the matrix start value:")
                        input2 = input("Enter the matrix end value:")
                        array_created = self.obj1.matrix_creation(input1, input2)
                        str1 = str(array_created)
                        # check output correct or not
                        if re.match(str1, 'None'):
                            print("Output will not display")
                        else:
                            # print("\nNew Matrix:\n", array_created)
                            print("\n 3 *4 Dimension matrix")

                            num1 = input("Enter the 1st dimension:")
                            num2 = input("Enter the 2nd dimension:")
                            result = self.obj1.reshape_matrix(array_created, num1, num2)
                            str2 = str(result)

                            if re.match(str2, 'None'):
                                print("Output will not display")
                            else:
                                print("Reshape given matrix into given dimension format: \n", result)
                                # multiply each element in array by 3
                                num = 3
                                print("Array  multiply every element value by 3: \n",
                                      self.obj1.matrix_scalar_multi(result, num))

                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        # 23
                        original_array = np.array([[0, 1], [2, 3], [4, 5]])
                        print("\nOriginal array: \n", original_array)
                        print("Array to list conversion:", original_array.tolist())
                        # 24
                        original_array1 = np.array([0.26153123, 0.52760141, 0.5718299, 0.5927067, 0.7831874, 0.69746349,
                                                    0.35399976, 0.99469633, 0.0694458, 0.54711478])
                        print("\nOriginal array: \n", original_array1)

                        np.set_printoptions(precision=3)
                        print("\nArray to list conversion with precision 3:", original_array1)

                        # 25
                        original_array2 = np.array([1.6e-10, 1.6, 1200, .235])
                        print("\nOriginal array: ", original_array2)
                        # np.set_printoptions(suppress=True)
                        np.set_printoptions(suppress=True)
                        print("Final array:", original_array2)

                    elif choice == 6:
                        # 26
                        inputarr = np.array([[10, 20, 30], [40, 50, 60]])
                        addarray = np.array([[100], [200]])
                        # np.append() add new coloumn to original array with axis = 1 for 2D array
                        print(np.append(inputarr, addarray, axis=1))

                    elif choice == 7:
                        # 27
                        array_input = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                        index = (0, 3, 4)
                        print(np.delete(array_input, index))
                    elif choice == 8:
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


# class Object created to call its methods
obj = ThreeDArray()
obj.calling()
