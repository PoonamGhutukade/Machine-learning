""" 26. Write a Python program to count the number occurrence of a specific character in a string. """

print()
str = "Hello Lila, How are you?"
print("Count of l is: ", str.count('l'))

print("-----------------OR---------------------------")

from collections import Counter

# count of each character
str1 = "Hello Lila, How are you?"
c = Counter(str1)
print("Count of each character: ")
for k, val in c.items():
    print(k, val)

