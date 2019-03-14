"""35. Write a Python program to get numbers divisible by fifteen from a list using an anonymous function. """

print("By normal method: Num divible by 15")
list1 = [10, 15, 20, 30, 45, 60, 50]
for num in list1:
    div = num % 15
    if div == 0:
        print(num)
    num = num + 1

print("-------------------Annonymuos_function---------------------------")
print()
# lambda keyword is used to create anonymous function
map_obj1 = map(lambda x: x % 15, list1)
map_obj2 = map(lambda x: x % 15 == 0, list1)
print(map_obj1)
# convert map object to a list
lis1 = list(map_obj1)
lis2 = list(map_obj2)
print("Using map: ", lis1)
print("Using map: ", lis2)
print()

print("--------------------------------------------------------")
obj2 = list(filter(lambda num: (num % 15 == 0), list1))
print("Using Filter: ", obj2)
print()
