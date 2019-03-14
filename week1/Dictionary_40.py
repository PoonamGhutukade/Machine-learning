""" 40. Write a Python program to extract single key-value pair of a dictionary in variables. """

dic = {"city": "mumbai"}

# with a tuple (just the comma):
(c1, c2), = dic.items()
# after bracket if comma is not given then it take only one value from dict and shows error
print(c1)
print(c2)

print()
print(type(dic))
print("--------------------------------")

# With list
[(c1, c2)] = dic.items()
print(c1)

print("--------------------------------")
# with iterate
c1, c2 = next(iter(dic.items()))
print(c1)
print(c2)
