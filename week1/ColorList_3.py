"""Write a Python program to display the first and last colors from the following list.
color_list = ["Red","Green","White" ,"Black"]
 """

color_list = ["Red","Green","White" ,"Black"]
print(color_list)

print("\nFirst Color: ",color_list[0],"--- Last Color: ",color_list[3])

print("-----------------------------------------------------------------------")

# Some extra functions.
print("\nFrom right to left 2nd element is: ",color_list[-2])

print("\nCut first two items")
print(color_list[0:2])


print("\nRemove Green")
color_list.remove("Green")
print(color_list)

print("\nUse pop():")
color_list.pop()
print(color_list)

print("\nUse del keyword:")
del color_list[0]
print(color_list)

