""" 4. Write a Python program to print the documents (syntax, description etc.) of Python built-in function(s).
Sample function : abs()
Expected Result :
abs(number) -> number
Return the absolute value of the argument."""

# Float Number
numb = -20.256
print("\nAbsolute value of number:", numb, " is: ", abs(numb))

# Integer Number
numb = -256
print("\nAbsolute value of number:", numb, " is: ", abs(numb))

# Complex Number
numb = 5 + 4j
print("\nAbsolute value of number:", numb, " is: ", abs(numb))

print("\nInfo: ", abs.__doc__)

print("--------------------------------------------------------------------------")

print("--------------------------------------------------------------------------")

"""
    @param 'a' as a by default variable with value 100
    show the details of abs() function
"""


def abs(a=100):
    """"This is a 
    user defined function"""

    return a


num = -50.20
print(abs(num).__doc__)
print()
print(abs.__doc__)
