""" 38. Write a Python program to add leading zeroes to a string. """

str2 = input("Enter data : ")
# ljust add zeroes at right side
# rjust add zeros at left size
print("\nWith ljust: ", str2.ljust(7, '0'))
print("With rjust: ", str2.rjust(7, '0'))


# If length is equal or less than(even if length is in minus) original string it gives original string only
print(str2.rjust(2, '0'))
print(str2.rjust(3, '0'))

# fillchar option is optional, it shows original string only
print()
print("Without fillchar: ", str2.rjust(10))
