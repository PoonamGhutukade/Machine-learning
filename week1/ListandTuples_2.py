""" 2. Write a Python program which accepts a sequence of comma-separated numbers from user and generate a list and a tuple with those numbers.
Sample data : 3, 5, 7, 23
Output :
List : ['3', ' 5', ' 7', ' 23']
Tuple : ('3', ' 5', ' 7', ' 23')
"""

# list and tuple from user
values = input("\nEnter Number for list: ")

Alist = values.split(",")
# maximum split is 2 for 2nd list
Alist1 = values.split(",", 2)
Btuple = tuple(Alist)

# Display list and tuple
print('List : ', Alist)
print('Tuple : ', Btuple)
print('List : ', Alist1)

# Access item
print("\nAccess item")
print("second item of list", Alist[1])
print("second item of tuple", Btuple[1])

# Change the item value
print("\nChange the second item of list and tuple")
Alist[1] = 10

# NewTuple[1]=100
print(Alist)
print("--Can't change tuple value, coz Tuples are unchangeable ")

print("\nLength of list", len(Alist))
print("Length of tuple", len(Btuple))

print("Add item 100 at 3rd position using insert")
Alist.insert(2, 100)
print(Alist)

print("Add item 800 using append")
Alist.append(800)
print(Alist)
