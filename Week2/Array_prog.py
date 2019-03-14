# Array -> It is similar to list, but in array all contents have similar type.We have to import array

from array import *
from Utility.Util import UtilClass
import re
""" Python does not have built-in support for Arrays, but Python Lists can be used instead."""


class ArrayProgram:
    # Utilityclass object is created to store and retrieve its values from that class
    obj1 = UtilClass()

    print("------------------------------------------")

    while True:
        try:
            print()
            print("1. Create an array of 5 integers and display ""\n""2. Reverse array"
                  "\n""3. Get the no. of occurrences of element in an array ""\n"
                  "4. Remove the first occurrence of a specified element""\n""5. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    # array created by taking input from user and show all elements in array with their index & values
                    result = obj1.createarr()
                    print("_______________________________________________________________________________")

                elif choice == 2:
                    # for integer use i
                    vals = array('i', [10, 22, 3, 45, 5, 3, 22])
                    print("\nOriginal array:", vals)
                    # for temp in range(5):
                    #     print(vals[temp])
                    # reverse the given array
                    array1 = obj1.reversearr(vals)
                    # print("Reverse array: ", array1)
                    print("_______________________________________________________________________________")

                elif choice == 3:
                    vals = array('i', [10, 22, 3, 45, 5, 3, 22])
                    # Check occurrences in array for specific element
                    obj1.countele(vals)
                    print("_______________________________________________________________________________")

                elif choice == 4:
                    vals = array('i', [10, 22, 3, 45, 5, 3, 22])
                    # Show array after removing 1st occurrence of specified element in array
                    obj1.removeele(vals)
                    print("_______________________________________________________________________________")

                elif choice == 5:
                    exit()
                    print("_______________________________________________________________________________")

                else:
                    print("Plz enter valid choice: ")

                acc = str(input("IF you want to continue: type yes "))
                # if acc == 'y' and acc == 'yes':
                if re.match(acc, 'y'):
                    continue
                elif re.match(acc, 'yes'):
                    continue
                elif re.match(acc, 'n'):
                    break
                elif re.match(acc, 'no'):
                    break
                else:
                    print("Give proper input")
                    continue

            else:
                raise ValueError
        except ValueError as e:
            print("\nInvalid Input")
