"""9. Write a Python program to append values to the end of an array.
Expected Output:
Original array:
[10, 20, 30]
After append values to the end of the array:
[10 20 30 40 50 60 70 80 90]"""

from Week4.Utility.Util import UtilClass
import numpy as np


class ArrayAppend:
    # class constructor
    def __init__(self):
        # list created
        self.list1 = [10, 20, 30]
        # utility class objected created here
        self.obj1 = UtilClass()

    def append_array(self):
        print("\nOriginal List:", self.list1)
        # OR return np.append(self.list1, [[40, 50, 60], [70, 80, 90]])
        # Syntax -> numpy.append(arr, values, axis)
        return np.append(self.list1, [40, 50, 60, 70, 80, 90], axis=0)


# class object created to call its methods
obj = ArrayAppend()

print("Append Array: ", obj.append_array())
