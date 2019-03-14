""" 20. Write a Python program to sort three integers without using conditional statements and loops. """
num1 = input("Enter 1st Number: ")
num2 = input("Enter 2nd Number: ")
num3 = input("Enter 3rd number: ")
num4 = int(input("Enter 4th number: "))
try:
    num1 = int(input("Enter 1st Number: "))
    num2 = int(input("Enter 2nd Number: "))
    num3 = int(input("Enter 3rd number: "))
    num4 = int(input("Enter 4th number: "))
    print()
    result1 = min(num1, num2, num3)
    result2 = max(num1, num2, num3)
    result3 = (num1 + num2 + num3) - result1 - result2

    print("Sorted 1st three numbers: ", result1, result3, result2)
except ValueError:
    print("Invalid")
print()



#---------------------------OR-------------------------------

list1=[num1, num2, num3, num4]
list1.sort()
print("Sorted List: ", *list1)
print("List: ", list1)