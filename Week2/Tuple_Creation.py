"""
    1. Write a Python program to create a tuple.
    2. Write a Python program to create a tuple with different data types.
"""


# tuple created with same data types
def tuplecreate():
    try:
        arrr = list()
        size = input("Enter the size of an tuple for same Dt:")
        # we have to typecast num to compare with length of string
        num2 = int(size)
        # checking enter value is only digit or not
        if size.isdigit():
            print("Enter the elements: ")
            for ele in range(num2):
                res = int(input())
                arrr.append(res)
                # typecast to tuple
                w = tuple(arrr)
            # print("Set Elements:", set(arrr))
            return w

        else:
            raise ValueError
    except ValueError:
        print("Enter valid number: ")


res = tuplecreate()
print("Tuple: ", res)


# tuple created with diff data types
def tuplecreatediffdatatypes():
    try:
        arrr = list()
        size = input("Enter the size of an tuple for Diff Dt:")
        # we have to typecast num to compare with length of string
        num2 = int(size)
        # checking enter value is only digit or not
        if size.isdigit():
            print("Enter the elements: ")
            for ele in range(num2):
                res = input()
                arrr.append(res)
                # typecast to tuple
                w = tuple(arrr)
            # print("Set Elements:", set(arrr))
            return w

        else:
            raise ValueError
    except ValueError:
        print("Enter valid number: ")


res = tuplecreatediffdatatypes()
print("Tuple: ", res)
