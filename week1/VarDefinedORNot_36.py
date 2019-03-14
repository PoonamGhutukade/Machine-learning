"""36. Write a Python program to determine if variable is defined or not. """
print()
# Use try except block to handled errors and exceptions
try:
    num = 1
except NameError:
    print("Variable not defined.....")
else:
    print("Variable defined.....")
finally:
    print("Finally Block")

print()
try:
    num2
except NameError:
    print("Variable not defined.....")
else:
    print("Variable defined.....")
finally:
    print("Finally Block")