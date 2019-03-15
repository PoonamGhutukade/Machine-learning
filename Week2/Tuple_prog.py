# tuples are immutable list is mutable
# from copy import deepcopy
# from collections import Counter
from Week2.Utility.Util import UtilClass
import re

"""
    1. Write a Python program to create a tuple. 
    2. Write a Python program to create a tuple with different data types. 
    3. Write a Python program to unpack a tuple in several variables. 
"""


class Tupleprog:
    obj = UtilClass()

    while True:
        try:
            print()
            print("1. create a tuple, with same & diff DT(Prog-1,2,3) ""\n""2. Create the colon of a tuple."
                  "\n""3. find the repeated items of a tuple ""\n"
                  "4. check whether an element exists within a tuple""\n"
                  "5. convert a list to a tuple ""\n"
                  "6. remove an item from a tuple""\n"
                  "7. slice a tuple""\n"
                  "8. reverse a tuple""\n"
                  "9. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    res = obj.creattuple()
                    # print("Int Tuple: ",res)
                    print("_______________________________________________________________________________")

                elif choice == 2:

                    """4. Write a Python program to create the colon(:) of a tuple. """

                    # create a tuple
                    tuplex = ("HELLO", 50, [], True)
                    obj.clonetuple(tuplex)

                    print("_______________________________________________________________________________")

                elif choice == 3:
                    """5. Write a Python program to find the repeated items of a tuple. """

                    tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7
                    coun = tupl.count(4)
                    print("\nOriginal tuple:", tupl)
                    print("Count of 4 is :", coun)
                    repitem = obj.show(tupl)
                    print("Repeated items: ", repitem)

                    print("_______________________________________________________________________________")

                elif choice == 4:

                    """6. Write a Python program to check whether an element exists within a tuple. """

                    tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7
                    print("Original tup: ", tupl)
                    print("Show element(7) exist in sys or not:")
                    obj.eleexist(tupl, 7)

                    print("_______________________________________________________________________________")

                elif choice == 5:
                    """7. Write a Python program to convert a list to a tuple. """
                    # llist = [15, 20, 66, 44, 99, 30]
                    res = obj.conlis()
                    print("Tuple: ", res)
                    print("_______________________________________________________________________________")

                elif choice == 6:

                    # You can not remove items from a tuple, but you can delete tuple completely
                    # It shoes UnboundLocalError

                    """8. Write a Python program to remove an item from a tuple. """
                    tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7
                    print("\nOriginal Tuple: ", tupl)
                    ret = obj.removeitem(tupl)
                    print("Tuple after removing items:")
                    print("_______________________________________________________________________________")

                elif choice == 7:

                    # slice tuple

                    """9. Write a Python program to slice a tuple.  """
                    tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7
                    print("\nOriginal Tuple: ", tupl)
                    obj.slicingtup(tupl)
                    print("_______________________________________________________________________________")

                elif choice == 8:
                    """10. Write a Python program to reverse a tuple. """

                    print("\nOriginal String", tupl)
                    print("Reverse string:", obj.reversetup(tupl))
                    print("-------------OR----------------")
                    aresult = tuple(reversed(tupl))
                    print("Rev Tuple with in built funct: ", aresult)
                    print("_______________________________________________________________________________")

                elif choice == 9:
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
                else:
                    print("Give proper input")
                    break

            else:
                raise ValueError
        except ValueError as e:
            print("\nInvalid Input")
