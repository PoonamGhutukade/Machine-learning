""" 41. Write a Python program to convert an integer to binary keep leading zeros.
Sample data : 50
Expected output : 00001100, 0000001100"""

print()
n = 10
print("Number: ", n)
print("\nDecimal to binary: ", bin(n))
print("Decimal to octal: ", oct(n))
print("Decimal to Hexa: ", hex(n))

print("\n8 - length size for binary: ", (format(n, '08b')))
print("10 - length size for binary: ", (format(n, '010b')))

print()
# if we want 0b as prefix
print(format(n, '#010b'))
