"""10. Write a Python program to find the real and imaginary parts of an array of complex numbers.
Expected Output:
Original array [ 1.00000000+0.j 0.70710678+0.70710678j]
Real part of the array:
[ 1. 0.70710678]
Imaginary part of the array:
[ 0. 0.70710678]
"""
import numpy as np


class ComplexClass:
    # directly give whole list otherwise find out sqrt of that complex number then show real and imaginary part
    # OR array1 = np.array([1.00000000+0.j,0.70710678+0.70710678j])
    array1 = np.sqrt([1 + 0.j])
    arr2 = np.sqrt([0 + 1.j])
    # append two array
    arr3v = array1 + arr2

    def display_real_and_imag(self):
        # show real part
        print("\nReal Part:", self.arr3v.real)
        # show imaginary part
        print("Imaginary part:", self.arr3v.imag)


# create object of out class
obj = ComplexClass()
obj.display_real_and_imag()
