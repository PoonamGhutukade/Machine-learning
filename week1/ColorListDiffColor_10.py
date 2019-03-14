"""
10. Write a Python program to print out a set containing all the colors from color_list_1 which are not present in color_list_2.
Test Data :
color_list_1 = set(["White", "Black", "Red"])
color_list_2 = set(["Red", "Green"])
Expected Output :
{'Black', 'White'}
"""

list_1 = {"red", "green", "blue"}

list_2 = {"blue", "white", "black"}
print("List1:", list_1)
print("List2", list_2)

print("\nUnion: ", list_1.union(list_2))

print('Difference: ', list_1.difference(list_2), "Not in 2nd list")

print('Intersection: ', list_1.intersection(list_2), "Common in both")
