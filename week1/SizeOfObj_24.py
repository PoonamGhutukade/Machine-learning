""" 24. Write a Python program to get the size of an object in bytes. """

import sys

str1 = "Hello"
str2 = "Good"
str3 = "Morning"
print()
print("Memory size of '"+str1+"' = "+str(sys.getsizeof(str1))+ " bytes")
print("Memory size of '"+str2+"' = "+str(sys.getsizeof(str2))+ " bytes")
print("Memory size of '"+str3+"' = "+str(sys.getsizeof(str3))+ " bytes")
print()

# By default for one length str size is 50 byte, and after that for each character it is incremented by one
n1 = 1
num1 = str(n1)
n2 = 20
num2 = str(n2)
n3 = 3000
num3 = str(n3)

print("Memory size of '"+num1+"' = "+str(sys.getsizeof(num1))+ " bytes ")
print("Memory size of '"+num2+"' = "+str(sys.getsizeof(num2))+ " bytes ")
print("Memory size of '"+num3+"' = "+str(sys.getsizeof(num3))+ " bytes ")

print()
print()
n1 = 10.125
num1 = str(n1)
n2 = 20.25
num2 = str(n2)
n3 = 30.50
num3 = str(n3)

print("Memory size of '"+num1+"' = "+str(sys.getsizeof(num1))+ " bytes ")
print("Memory size of '"+num2+"' = "+str(sys.getsizeof(num2))+ " bytes ")
print("Memory size of '"+num3+"' = "+str(sys.getsizeof(num3))+ " bytes ")