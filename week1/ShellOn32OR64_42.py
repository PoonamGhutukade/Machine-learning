""" 42. Write a Python program to determine if the python shell is executing in 3
2bit or 64bit mode on operating system. """

import struct

# struct from std.library
print(struct.calcsize("P") * 8)

print("-------------------OR-----------------")

import ctypes

print(ctypes.sizeof(ctypes.c_void_p))
# It will be 4 for 32 bit or 8 for 64 bit.

print("-------------------OR-----------------")

import platform

print(platform.architecture()[0])
print(platform.architecture()[1])
print(platform.architecture())
# architecture(bits,linkage format for executable)
print("-------------------OR-----------------")

import sys

print(sys.maxsize > 2 ** 32)
# display true  if sys is 64 bit otherwise display false

print("-------------------OR-----------------")
import struct

# struct from std.library
print(struct.calcsize("P") * 8)

print("-------------------OR-----------------")