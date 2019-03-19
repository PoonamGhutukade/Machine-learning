"""1. Write a Python program to convert a list of numeric value into a one-dimensional NumPy array.
Expected Output:
Original List: [12.23, 13.32, 100, 36.32]
One-dimensional numpy array: [ 12.23 13.32 100. 36.32]"""
from Week4.Utility.Util import UtilClass


class NumpyClass1:
    # class constructor
    def __init__(self):
        self.original_list = [12.23, 13.32, 100, 36.32]
        self.obj1 = UtilClass()

    def calling(self):
        print("\nOriginal List:", self.original_list)
        print("\nList to ndarray conversion:", self.obj1.convert_list_ndarray(self.original_list))


# class object created to call its methods
obj = NumpyClass1()
obj.calling()
