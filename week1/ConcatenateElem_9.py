"""Write a Python program to concatenate all elements in a list into a string and return it."""

print()
# define all num as string
list1 = ['10', '20', '3', '5']
lis1 = "".join(list1)
print(lis1)
print()

print("---------------------OR--------------------------")
# If we taking number in a list
Alist = [1, 20, 3, 4]
lis1 = ''.join(map(str, Alist))
print(lis1)
print()

print("---------------------OR--------------------------")
"""
    @param list
    To Concatenate all elements in a list
"""
def concate(list):
    result = ''
    for num in list:
        # typecasting is required
        result += str(num)
    return result


print(concate([2, 4, 6, 8]))
