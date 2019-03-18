"""13. The table below shows the height, x, in inches and the pulse rate, y, per minute,
for 9 people. Write a program to find the correlation coefficient and interpret your result.
x ⇒ 68 72 65 70 62 75 78 64 68
y ⇒ 90 85 88 100 105 98 70 65 72"""

# Python Program to find correlation coefficient.
import math


# function that returns correlation coefficient.
def correlation_coefficient(X, Y, n):
    sum_of_x = 0
    sum_of_y = 0
    sum_of_xy = 0
    square_sum_x = 0
    square_sum_y = 0

    # Check for all values in list
    i = 0
    while i < n:
        # sum of elements of array X.
        sum_of_x = sum_of_x + X[i]

        # sum of elements of array Y.
        sum_of_y = sum_of_y + Y[i]

        # sum of X[i] * Y[i].
        sum_of_xy = sum_of_xy + X[i] * Y[i]

        # sum of square of array elements.
        square_sum_x = square_sum_x + X[i] * X[i]
        square_sum_y = square_sum_y + Y[i] * Y[i]

        i = i + 1

    # use formula for calculating correlation coefficient.
    corr = float(n * sum_of_xy - sum_of_x * sum_of_y) / float(
    math.sqrt((n * square_sum_x - sum_of_x * sum_of_x) * (n * square_sum_y - sum_of_y * sum_of_y)))

    return corr


"""
height, x, in inches
the pulse rate, y, per minute,
for 9 people i.e. Total Count = 9
"""
# Driver function
X = [68, 72, 65, 70, 62, 75, 78, 64, 68]
Y = [90, 85, 88, 100, 105, 98, 70, 65, 72]

# Find the size of array.
n = len(X)

# Function call to correlationCoefficient.
print("\ncorrelation coefficient:", '{0:.6f}'.format(correlation_coefficient(X, Y, n)))
