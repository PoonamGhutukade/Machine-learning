"""Write a Python program to create a histogram from a given list of integers. """

print("By User defined function")
"""
    @param num
    create a histogram 
"""


def histogram(list1):
    print(list1)
    for num in list1:
        show = " "
        times = num
        while (times > 0):
            show += "*"
            times = times - 1

        print(show)


# use list to show histogram
histogram([1, 2, 3, 4])
print()
print("---------------------------------OR------------------------------------------------")

value1 = input("Enter values for creating histogram: ").split(',')
print(value1)

for n11 in value1:
    # typecasting is required when we take user input
    num1 = int(n11)
    # we use ** for power of number
    print(num1 * "$" * num1)
print()

print("----------------------------------OR-----------------------------------------------")
print("Another Method:")
arr = 2, 3, 4, 5, 6
temp = "*"
for temp1 in arr:
    print(temp1 * temp)

print("----------------------------------OR-----------------------------------------------")
for temp in range(4):
    for temp1 in range(4 - temp):
        # end= "" to print content on same line
        print("#", end=" ")
    # to take next line
    print()
