"""Write a Python program which accepts the user's first and last name
and print them in reverse order with a space between them."""

fname = input("Enter your first name: ")
lname = input("Last name: ")

# validation for string
try:
    if fname.isalpha():
        rest = lname + " " + fname
        print("Hello " + rest)
    else:
        raise ValueError
except ValueError:
    print("invalid Input, enter string only")

print("-----------------------OR----------------------")

"""
    @param str
    user defined function Rev to reverse the string
"""


def reverse(string):
    string = string[::-1]
    return string


fname = input("Enter your first name: ")
lname = input("Last name: ")
# Validation : Exception handling
try:
    if fname.isalpha():
        str = fname + " " + lname
        print("Reverse string: ", reverse(str))
    else:
        raise ValueError
except ValueError:
    print("Plz enter valid input as string only")

