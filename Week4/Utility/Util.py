import numpy as np
import numpy.matlib
import pandas as pd
import math


class UtilClass:

    def convert_list_ndarray(self, original_list):
        # asarray is useful to converting python sequence into ndarray
        return np.asarray(original_list)

    # create new matrix
    def matrix_creation(self, input1, input2):
        try:
            # input1 = input("\nEnter the matrix start value:")
            # input2 = input("Enter the matrix end value:")
            start = int(input1)
            end = int(input2)
            if input1.isdigit() and input2.isdigit():
                # It display number from start to end input value
                return np.arange(start, end + 1)
            else:
                raise ValueError
        except Exception as e:
            print("\nInvalid Input:", e)

    # reshape created matrix
    def reshape_matrix(self, array1, num1, num2):
        try:
            first_dimension = int(num1)
            second_dimension = int(num2)
            if num1.isdigit() and num2.isdigit():
                # reshape array
                final_aaray = array1.reshape(first_dimension, second_dimension)
                # check datatype of given array, by default datatype is int64 for integer
                print("Array DT: ", np.array(final_aaray).dtype)
                # this return the number of array dimension
                print("\nArray Dimension for following array is :", final_aaray.ndim)

                return final_aaray
            else:
                raise ValueError
        except Exception as es:
            print("cannot reshape array :", es)

    def null_vector_creation(self, array2):
        # np.zeroes allow us to create array with zero fields
        array3 = (len(array2))
        # use nympy.matlib library for matrix operations
        # OR array4 = np.matrix.zeros(array3)
        return np.matlib.zeros(array3)

    def update_matrix(self, array4):
        # print("\nOriginal null vector array of size 10:", array4)
        # to update array value, syntax -> numpy.put(array, index, value, mode='raise')
        np.put(array4, 6, 11)
        return array4

    # reverse matrix method
    def matrix_reverse(self, array_created):
        return array_created[::-1]

    def matrix_one_creation(self, array2):
        # np.zeroes allow us to create array with zero fields
        array3 = (len(array2))
        return np.matlib.ones(array3)

    """find the number of elements of an array, length of one array element in bytes and
          total bytes consumed by the elements"""

    def array_size_ele(self):
        # OR x = np.array([1,2,3], dtype=np.float64)
        """
        for int8 -> one element = 1 byte('i1')
        for int16 -> one element = 2 bytes('i2')
        for int32 -> one element = 4 bytes('i4')
        for int64 -> one element = 8 bytes('i8')
        """
        # OR x = np.array([1, 2, 3, 4, 5], dtype = np.int64)
        x = np.array([1, 2, 3, 4, 5], np.dtype('i8'))
        # find the number of elements of an array
        print("Size of the array: ", x.size)
        # length of one array element in bytes
        print("Length of one array element in bytes: ", x.itemsize)
        #  bytes consumed by the elements
        print("Total bytes consumed by the elements of the array: ", x.nbytes)

    """find common values between two arrays
        find the set difference of two arrays
        Set exclusive-or will return the sorted, unique values """

    def operation_on_two_arrays(self):
        array1 = [0, 10, 20, 40, 60, 80]
        array2 = [10, 30, 40, 50, 70, 90]
        print("\nOriginal arr1: ", array1)
        print("Original arr2: ", array2)

        common = np.intersect1d(array1, array2)
        print("\nCommon from both lists: ", common)
        diff = np.setdiff1d(array1, array2)
        print("\nDiffrence  from both lists: ", diff)
        diff = np.setxor1d(array1, array2)
        print("\nUnique values that are in only one from both lists: ", diff)
        diff = np.union1d(array1, array2)
        print("\nUnion from both lists: ", diff)

    def compare_two_array(self):
        arr1 = [1, 2]
        arr2 = [3, 4]
        print("array1: ", arr1)
        print("array2: ", arr2)
        # check 2 arrays for less or greater condition
        print("arr1 > arr2: ", np.greater(arr1, arr2))
        print("arr1 >= arr2: ", np.greater_equal(arr1, arr2))
        print("arr1 < arr2 ", np.less(arr1, arr2))
        print("arr1 <= arr2: ", np.less_equal(arr1, arr2))

    def flattendarr(self, arr1):
        arr = np.ndarray.flatten(arr1)
        print("Dimension for below array :", arr.ndim)
        # convert nD array to 1D array
        return np.ndarray.flatten(arr1)

    """
            for float8 -> one element = 1 byte('f1')
            for float16 -> one element = 2 bytes('f2')
            for float32 -> one element = 4 bytes('f4')
            for float64 -> one element = 8 bytes('f8')
    """

    def change_dt(self, arr12):
        # change datatype int to float64
        return np.array(arr12, dtype='f8')

    def three_d_arr(self):
        output = self.reshape_matrix(self.matrix_creation())
        arr_new = np.array(output, ndmin=3)
        # print("3D Array: \n", arr_new)
        return arr_new

    def identitymatrix(self, arr):
        arr1 = len(arr)
        # np.matlib.eye only take integer number for array, so have to find out length of given array
        identity_matrix = np.matlib.eye(arr1, dtype=float)
        final_matrix = np.array(identity_matrix, ndmin=3)
        print("Dimension for below matrix: ", final_matrix.ndim)
        return final_matrix

    def concreate_data(self, data1, data2):
        # concatenate two array
        return np.concatenate((data1, data2), 1)

    # array multiply with each element by 3
    @staticmethod
    def matrix_scalar_multi(array, num):
        for temp in np.nditer(array, op_flags=['readwrite']): temp[...] = num * temp
        return array
# __________________________________________________________________________________________________________________
    # pandas functions

    # create panda series
    def create_series(self, size):
        try:
            arrr = list()
            # we have to typecast num to compare with length of string
            num2 = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter the elements: ")
                for ele in range(num2):
                    res = int(input())
                    arrr.append(res)
                    # put list into panda series
                panda_series = pd.Series(arrr)
                return panda_series

            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    # conver panda_series to list
    def conversion(self, panda_series1):
        print(type(panda_series1.tolist()))
        return panda_series1.tolist()

#
    def series_operations(self, series1, series2):
        add = series1 + series2
        print("\nAddition of two series\n", add)
        sub = series1 - series2
        print("Substraction of two series\n", sub)
        mul = series1 * series2
        print("Multiplication of two series\n", mul)
        div = series1 / series2
        print("Division of two series\n", div)

    def series_power(self, panda_series):
        result = []
        for temp in panda_series:
            res = math.pow(temp, 2)
            result.append(res)
        return result
