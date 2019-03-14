"""  Write a Python program to check whether a specified value is contained in a group of values.
Test Data :
3 -> [1, 5, 8, 3] : True
-1 -> [1, 5, 8, 3] : False
"""

"""
    @param group and num1
    check whether a specified value is contained in a group of values
"""


def is_member(group, num1):
    for val in group:
        if num1 == val:
            return True
    return False


print("One Method")
print(is_member([1, 5, 8, 3], 3))
print(is_member([1, 5, 8, 3], -1))
print()
# ------------------------------------------------------------------------

"""
    @param num
    check whether a specified value is contained in a group of values
"""


def is_present(num):
    arr = [1, 2, 3, 4]
    return num in arr


print("Another Way")
print(is_present(3))
print(is_present(-1))
# ----------------------------------------------------------------------------

# search value by user
values = ['10', '20', '30', '40']
print()
print(values)
print(input("Enter number to search: ") in values)
