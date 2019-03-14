"""  17. Write a program to get execution time for a Python method. """

import time

print()

"""
    @param num
    sum() Calculate addition of numbers and
    Calculate the execution time 
"""


def addition(num):
    starttime = time.time();
    summ = 0
    # for range values python by default take num-1 for num
    for n in range(1, num + 1):
        summ += int(n);
    endTime = time.time()
    return summ, endTime - starttime


# num = input("Enter number for addition").split(",")
num = 5
print("Addition from 1 to ", num, "Requires time is: ", addition(num))

print("-------------------------------OR----------------------------------")
# another way to calculate time
print()
print("Addition:")
stime = time.time()
num = int(input("Enter the number for finding addition till that number: "))
print("Addition: ", num * (num + 1) / 2)
etime = time.time()
print("Executipn time: ", etime - stime)
