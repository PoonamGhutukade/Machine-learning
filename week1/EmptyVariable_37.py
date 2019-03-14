""" Write a Python program to empty a variable without destroying it.
Sample data: n=20
d = {"x":200}
Expected Output : 0
{}
"""
n = 10
dic = {"x": 10}
lis = [10, 20, 30]
tup = (10, 20, 3, 4)
z = None
print(type(z))
print(type(n))
print(type(dic))
print(type(lis))
print(type(tup))
print()

# empty a variable
print(type(n)())
print(type(dic)())
print(type(lis)())
print(type(tup)())
print(type(z)())
del z
# If we check for following line, it gives error as z is not defined
# print(type(z))
# del keyword is used to destroy the object
