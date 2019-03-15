"""Strings"""
#
# import textwrap
# from collections import Counter
import re
from Week2.Utility.Util import UtilClass


class StringProg:
    obj = UtilClass()

    while True:
        try:
            print()
            print("1. Calculate the length of a string ""\n""2. count the number of characters"
                  "\n""3. first char have been changed to '$' ""\n"
                  "4. add 'ing' or 'ly at the end of a given string""\n"
                  "5. convert a list to a tuple ""\n"
                  "6. Display input back in upper and lower cases.""\n"
                  "7. prints the unique words in sorted form""\n"
                  "8. get the last part of a string before a specified character""\n"
                  "9. display formatted text (width=50) as output.""\n""10. count occurrences of a substring""\n"
                  "11. reverse a string""\n""12. lowercase first n characters in a string""\n"
                  "13. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    """1. Write a Python program to calculate the length of a string."""

                    result = obj.strlen()
                    res = str(result)
                    print("_______________________________________________________________________________")

                elif choice == 2:
                    """2. Write a Python program to count the number of characters (character frequency) in a string."""
                    # print("\n Count character in string: ")
                    # obj.countcharinstr(res)

                    result14 = obj.strlen()
                    wordab = str(result14)

                    if re.match(wordab, 'None'):
                        print("Empty set, Create again:")
                        result11 = obj.strlen()
                        wordab1 = str(result11)
                        if re.match(wordab1, 'None'):
                            continue
                        else:
                            print("\nOriginal Set: ", result11)
                            print("\n Count character in string: ")
                            obj.countcharinstr(result11)

                    else:
                        print("\nOriginal Set: ", result14)
                        print("\n Count character in string: ")
                        obj.countcharinstr(result14)

                    print("_______________________________________________________________________________")

                elif choice == 3:
                    """3. Write a Python program to get a string from a given string where all occurrences of its 
                        first char have been changed to '$', except the first char itself. """

                    result15 = obj.strlen()
                    wordab = str(result15)

                    if re.match(wordab, 'None'):
                        print("Empty set, Create again:")
                        result114 = obj.strlen()
                        wordabc = str(result15)

                        if re.match(wordabc, 'None'):
                            continue
                        else:
                            print("\nOriginal String: ", wordabc)
                            res1212 = obj.strtodolor(wordabc)
                            print("\nString with $: ", res1212)
                        continue

                    else:
                        sample = str(result15)
                        print("\nOriginal String: ", sample)
                        res12 = obj.strtodolor(sample)
                        print("\nString with $: ", res12)

                    print("_______________________________________________________________________________")

                elif choice == 4:
                    """
                        4. Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
                        If the given string already ends with 'ing' then add 'ly' instead. I
                        f the string length of the given string is less than 3, leave it unchanged.
                        Sample String : 'abc'
                        Expected Result : 'abcing'
                        Sample String : 'string'
                        Expected Result : 'stringly'
                        """

                    result155 = obj.strlen()
                    wordab = str(result155)

                    if re.match(wordab, 'None'):
                        print("Empty set, Create again:")
                        result1122 = obj.strlen()
                        wordab18 = str(result1122)
                        if re.match(wordab18, 'None'):
                            continue
                        else:
                            obj.adding(result1122)

                    else:
                        obj.adding(result155)
                    print("_______________________________________________________________________________")

                elif choice == 5:
                    """7. Write a Python program to convert a list to a tuple. """
                    # llist = [15, 20, 66, 44, 99, 30]
                    res = obj.conlis()
                    print("Tuple: ", res)
                    print("_______________________________________________________________________________")

                elif choice == 6:
                    """6. Write a Python script that takes input from the user and displays that input back in upper 
                    and lower cases."""

                    result156 = obj.strlen()
                    wordab12 = str(result156)

                    if re.match(wordab12, 'None'):
                        print("Empty set, Create again:")
                        result11234 = obj.strlen()

                        wordab189 = str(result11234)

                        if re.match(wordab189, 'None'):
                            continue
                        else:
                            obj.adding(result11234)

                    else:
                        obj.upperlower(result156)

                    print("_______________________________________________________________________________")

                elif choice == 7:
                    """
                        7. Write a Python program that accepts a comma separated sequen
                        ce of words as input and prints
                         the unique words in sorted form (alphanumerically).
                        Sample Words : red, white, black, red, green, black
                        Expected Result : black, green, red, white,red
                        """
                    out = obj.sortstrs()
                    print("\nSorted list: ", out)
                    print("_______________________________________________________________________________")

                elif choice == 8:
                    """8. Write a Python program to get the last part of a string before a specified character. """
                    strnew = 'https://www.ex.com/python-abcd-1234'
                    print("Original String-> ", strnew)
                    obj.lastpartstr(strnew)
                    print("_______________________________________________________________________________")

                elif choice == 9:
                    """9. Write a Python program to display formatted text (width=50) as output."""

                    text = "Hello, Trees usually reproduce using seeds. Flowers and fruit may be present, " \
                           "but some trees, such as conifers, instead have pollen cones and seed cones. Palms, bananas,"\
                           " and bamboos also produce seeds, but tree ferns produce spores instead."
                    print("\nFormatted text as width=50")
                    resl6 = obj.formatetxt(text, )
                    print(resl6)
                    print("_______________________________________________________________________________")
                elif choice == 10:
                    """10. Write a Python program to count occurrences of a substring in a string."""

                    strn = 'Hello, i am a latest desktop.thank You desktop family'
                    outp = obj.countsubstr(strn, 'desktop')
                    print("\nOccerences of substr is:", outp)

                    print("_______________________________________________________________________________")

                elif choice == 11:
                    """11. Write a Python program to reverse a string."""

                    result19 = obj.strlen()
                    wordabc = str(result19)

                    if re.match(wordabc, 'None'):
                        print("Empty set, Create again:")
                        result19 = obj.strlen()
                        wordab19 = str(result19)

                        if re.match(wordab19, 'None'):
                            continue
                        else:
                            res12123 = obj.reverse11(result19)
                            print("\nReverse string: ", res12123)

                    else:
                        res1233 = obj.reverse11(result19)
                        print("\nReverse string: ", res1233)

                    print("_______________________________________________________________________________")

                elif choice == 12:
                    """12. Write a Python program to lowercase first n characters in a string."""

                    str9 = 'POONAM'
                    obj.func(str9)
                    print("_______________________________________________________________________________")

                elif choice == 13:
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
                    break

            else:
                raise ValueError
        except ValueError as e:
            print("\nInvalid Input")
