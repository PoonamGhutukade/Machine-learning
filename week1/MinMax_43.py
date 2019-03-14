"""  Write a Python function to find the maximum and minimum numbers from a sequence of numbers.
Note: Do not use built-in functions.
"""

"""
    @param list
    Find out min and max number from list
"""


def min_max_func(list):
    large = list[0]
    small = list[0]
    for num in list:
        if num > large:
            large = num
        elif num < small:
            small = num

    return "Large Num::", large, "Small Num:", small


print(min_max_func([10, 5, 19, 85, 20, 4, 66, 70]))
