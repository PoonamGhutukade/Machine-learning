"""Sets1. Write a Python program to create a set.
2. Write a Python program to iteration over sets.
3. Write a Python program to add member(s) in a set.
4. Write a Python program to remove item(s) from set
5. Write a Python program to remove an item from a set if it is present in the set.
6. Write a Python program to create an intersection of sets.
7. Write a Python program to create a union of sets.
8. Write a Python program to create set difference.
9. Write a Python program to create a symmetric difference.
10. Write a Python program to clear a set.
11. Write a Python program to use of frozensets.
12. Write a Python program to find maximum and the minimum value in a set.
"""
import re
from Week2.Utility.Util import UtilClass


class SetProg:
    obj = UtilClass()
    while True:
        try:
            print()
            print("1. create a set  ""\n""2. iteration over sets."
                  "\n""3. add member(s) in a set ""\n"
                  "4. remove item(s) from set""\n"
                  "5. create an intersection of sets ""\n"
                  "6. create a union of sets""\n"
                  "7. create set difference.""\n"
                  "8. create a symmetric difference""\n"
                  "9. clear a set.""\n"
                  "10. use of frozensets""\n"
                  "11. find maximum and the minimum value in a set""\n""12. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    # empty set
                    result1 = obj.setcreation()
                    print("--------OR------------")
                    result1111 = obj.crestesetbyuser()
                    print("Set: ", result1111)
                    print("_______________________________________________________________________________")

                elif choice == 2:
                    # 2nd prog
                    result1111 = obj.crestesetbyuser()
                    w = str(result1111)
                    if len(w) == 0:
                        print("create list")
                        obj.crestesetbyuser()
                    elif re.match(w, 'None'):
                        # If set is empty create again and then show its iteration again
                        print("Empty set, Create again:")
                        newval = obj.crestesetbyuser()
                        continue
                        # print("\nShow iteration over set")
                        # resq = obj.iterforset(newval)
                        # for temp in resq:
                        #     print(temp)
                    else:
                        print("\nShow iteration over set")
                        resd = obj.iterforset(result1111)
                        for temp in resd:
                            print(temp)

                    print("_______________________________________________________________________________")

                elif choice == 3:
                    # 3rd prog

                    result14 = obj.crestesetbyuser()
                    wordab = str(result14)

                    if re.match(wordab, 'None'):
                        # If set is empty create again and then show its iteration again
                        print("Empty set, Create again:")
                        newval = obj.crestesetbyuser()
                        continue
                        # print("\nOriginal Set: ", result1)
                        # res = obj.addinset(result1)
                        # print("\nSet after adding elements:", res)
                    else:
                        print("\nOriginal Set: ", result14)
                        res = obj.addinset(result14)
                        print("\nSet after adding elements:", res)

                    print("_______________________________________________________________________________")

                elif choice == 4:
                    # 4th and 5th
                    set18 = {'Red', 'Green', 'White', 'Black', 'Grey', 'Yellow', 'orange', 'gery'}
                    resl = obj.remitems(set18)
                    print("after removing items which is not present: ", resl)

                    print("_______________________________________________________________________________")

                elif choice == 5:
                    # 6th
                    setz = {1, 2, 3, 4, 5}
                    setc = {3, 4, 5, 6, 7}
                    print("\nSet1: ", setz)
                    print("Set2: ", setc)
                    obj.intersectionset(setz, setc)

                    print("_______________________________________________________________________________")

                elif choice == 6:
                    # 7th
                    setz = {1, 2, 3, 4, 5}
                    setc = {3, 4, 5, 6, 7}
                    print("\nSet1: ", setz)
                    print("Set2: ", setc)
                    obj.unionset(setz, setc)

                    print("_______________________________________________________________________________")

                elif choice == 7:
                    # 8th
                    print("\nSet1: ", setz)
                    print("Set2: ", setc)
                    obj.setdiff(setz, setc)
                    print("_______________________________________________________________________________")

                elif choice == 8:
                    # 9th
                    print("\nSet1: ", setz)
                    print("Set2: ", setc)
                    obj.setsymmetricdiff(setz, setc)

                    print("_______________________________________________________________________________")

                elif choice == 9:
                    # 10th Clear set
                    setz = {1: 10, 2: 20, 3: 30}
                    obj.setclear(setz)
                    print("_______________________________________________________________________________")

                elif choice == 10:
                    # 11th
                    obj.usefrozenset()
                    print("_______________________________________________________________________________")

                elif choice == 11:
                    setz = {1: 10, 2: 20, 3: 30}
                    obj.minmax(setz.values())
                    print("_______________________________________________________________________________")

                elif choice == 12:
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
