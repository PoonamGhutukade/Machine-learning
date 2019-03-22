"""8. Write a Python program to convert a list and tuple into arrays."""

from Week4.Utility.Util import UtilClass
import numpy as np


class ConversionOfListTuple:
    # class constructor
    def __init__(self):
        # list created
        self.list1 = [1, 2, 3, 4, 5, 6, 7, 8]
        # tuple created
        self.tuple1 = ((8, 4, 6), (1, 2, 3))
        # utility class objected created here
        self.obj1 = UtilClass()

    def list_to_array(self):
        print("\nOriginal List:", self.list1)
        return np.array(self.list1)

    def tuple_to_array(self):
        print("\nOriginal tuple:", self.tuple1)
        return np.array(self.tuple1)


# class object created to call its methods
obj = ConversionOfListTuple()

print("List to array conversion: ", obj.list_to_array())
print("Tuple to array conversion: \n", obj.tuple_to_array())
