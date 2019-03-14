
import re
from Utility.Util import UtilClass


class ListOpertn:
    # class or static list
    obj = UtilClass()

    while True:
        try:
            print()
            print("1. sum all the items in a list  ""\n""2. multiplies all the items in a list."
                  "\n""3. get the smallest number from a list ""\n"
                  "4. count the number of strings""\n"
                  "5. sorted in increasing order ""\n"
                  "6. remove duplicates from a list""\n"
                  "7. clone or copy a list""\n"
                  "8. find the list of words that are longer""\n"
                  "9. takes two lists and returns True for common item.""\n"
                  "10. Print a specified list after removing the 0th, 4th and 5th elements""\n"
                  "11. generate all permutations of a list""\n"
                  "12. get the difference between the two lists""\n"
                  "13. append a list to the second list""\n"
                  "14. check whether two lists are circularly identical""\n"
                  "15. find common items from two lists""\n"
                  "16. split a list based on first character of word""\n"
                  "17. remove duplicates from a list of lists""\n"
                  "18. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    # By for loop
                    z = obj.add()
                    print("Addition of all elements in list: ", z)
                    print("----------------OR-----------------")
                    # using lambda function
                    print("Sum of all ", obj.additn())
                    print("_______________________________________________________________________________")

                elif choice == 2:
                    print("Multiplication of all elements: ", obj.multiplication())
                    print("_______________________________________________________________________________")

                elif choice == 3:
                    obj.min_max()

                    print("_______________________________________________________________________________")

                elif choice == 4:
                    list1 = ['abc', 'xyz', 'aba', '1221']
                    print("Original String: ", )
                    obj.countchar(list1)
                    print("_______________________________________________________________________________")

                elif choice == 5:
                    list12 = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
                    print("\nOriginal List: ", list12)
                    print("\nSorted list: ")
                    print(obj.sortlist(list12))
                    print("_______________________________________________________________________________")

                elif choice == 6:
                    sample_List = ['Red', 'Green', 'White', 'Red', 'Pink', 'Red']
                    obj.withoutduplicate(sample_List)

                    print("_______________________________________________________________________________")

                elif choice == 7:
                    listcply = [2, 5, 1, 4, 3, 2]
                    obj.copylist(listcply)

                    print("_______________________________________________________________________________")

                elif choice == 8:
                    list11 = ['hello', 'cat', 'good', 'oh', 'beautiful', 'nice']
                    num = input("Enter length for display list elements: ")
                    print("Words whose length is less than or equal to ", num)
                    obj.strlen(list11, num)
                    print("_______________________________________________________________________________")

                elif choice == 9:
                    # find out common in tow list another one is in function
                    slist1 = ['Red', 'Green', 'White', 'Red', 'Pink', 'Red']
                    # filter create filter object
                    filobj = filter(obj.usefilter, slist1)

                    # use dic to print only unique elements and convert filter object into list
                    result = list(dict.fromkeys(filobj))
                    lislen = len(result)
                    print()
                    print("\nIs there is any common element:")
                    if lislen > 0:
                        print("True")
                    else:
                        print("False")
                    print("_______________________________________________________________________________")

                elif choice == 10:
                    print()
                    samplelist12 = ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
                    print("\nOriginal list: ", samplelist12)
                    print("Remaining elements:")
                    obj.skipcolor(samplelist12)
                    print("_______________________________________________________________________________")

                elif choice == 11:
                    print()
                    listt = [1, 2, 3]
                    print("Permunation list for :", listt)
                    obj.permu(listt)
                    print("---------------OR--------------")
                    list222 = ['A', 'B', 'C', 'D']
                    # ll=  [1,2,3]
                    print("Permunation list for :", list222)
                    for value in obj.all_perms(list222):
                        print(value)
                    print("_______________________________________________________________________________")

                elif choice == 12:
                    print()
                    list1 = [1, 2, 3, 4, 8, 9]
                    list2 = [2, 3, 4, 5, 6, 7]
                    print("\nOriginal List:", list1, list2)
                    result = obj.difference(list1, list2)
                    print("Diff bet two list", result)
                    print("_______________________________________________________________________________")

                elif choice == 13:
                    print()
                    stack1 = ['a', 'b', 'c']
                    stack2 = ['d', 'e', 'f']
                    print("Append two list:")
                    result111 = obj.appendlist1(stack1, stack2)
                    print(result111)
                    print("------------OR-----------------------")
                    # Using Extend
                    list11 = ['a', 'b', 'c']
                    list22 = ['d', 'e', 'f']
                    print("Extend two list:")
                    result222 = obj.appendlist23(list11, list22)
                    print(result222)
                    print("_______________________________________________________________________________")

                elif choice == 14:
                    list1 = [10, 10, 0, 0, 10]
                    list2 = [10, 10, 0, 0, 10]
                    print("\nIS List Circular Identical: ")
                    obj.ciridenti(list1, list2)
                    print("_______________________________________________________________________________")

                elif choice == 15:
                    lista = [1, 2, 3, 4, 8, 9]
                    listb = [2, 3, 4, 5, 6, 7]
                    obj.listcommonitem(lista, listb)
                    print("_______________________________________________________________________________")

                elif choice == 16:
                    exlist = ['About', 'Absolutely', 'After', 'All', 'Also', 'Amos', 'And', 'Anyhow',
                              'Are', 'As', 'At', 'Aunt', 'Aw', 'By', 'Behind', 'Besides', 'Biblical', 'Bill', 'Bye']
                    print("\nshow all words by letter")
                    obj.spitlistbyword(exlist)
                    print("_______________________________________________________________________________")

                elif choice == 17:
                    slist = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
                    obj.removeduplicate(slist)
                    print("_______________________________________________________________________________")

                elif choice == 18:
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













