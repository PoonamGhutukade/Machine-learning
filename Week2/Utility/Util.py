from functools import reduce
from array import *
# import copy for deepcopy to copy tuple
import copy
# import counter to count item in tuple
from collections import Counter
import itertools
from itertools import permutations
from itertools import groupby
from operator import itemgetter
import textwrap


class UtilClass:
    # print("-------------------------------ARRAY---------------------------------------------------------------")
    """ 1. Write a Python program to create an array of 5 integers and display the array items.
    Access individual element through indexes.
    """

    # array created by taking input from user
    def createarr(self):
        # global array variable for array creation method
        global ar
        try:
            arrr = list()
            size = input("Enter the size of an array:")
            # we have to typecast num to compare with length of string
            num2 = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter the elements: ")
                for ele in range(num2):
                    res = int(input())
                    arrr.append(res)
                    # in array i is -> signed integer, f-> float(size 4 byte), d ->float(size 8)
                    ar = array('i', arrr)
                print("Array Elements:", ar)

            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

        cont = 0
        # show all elements in array with their index & values
        for temp in arrr:
            print("Index:", cont, "element:", temp)
            cont += 1
        print()

    """ 2. Write a Python program to reverse the order of the items in the array. """

    # reverse the given array
    def reversearr(self, array1):
        # reverse
        array1.reverse()
        print("Reverse array: ", array1)

    """ 3. Write a Python program to get the number of occurrences of a specified element in an array.  """

    # Check occurrences in array for specific element
    def countele(self, vals):
        print("Original Array: ", vals)
        print("Count of element 3 is: ", vals.count(3))

    """4. Write a Python program to remove the first occurrence of a specified element from an array. """

    # Show array after removing 1st occurrence of specified element in array
    def removeele(self, vals):
        print("Original Array: ", vals)
        print("Array after removing 1st occurrence of 22:")
        # remove 22 from given array
        vals.remove(22)
        print(vals)

    # print("---------------------------------------Dictionary--------------------------------------------------")
    """ 1. Write a Python script to sort (ascending and descending) a dictionary by value. """

    # show dic in asce and desc order by their values
    def asecdec(self, dic):
        asec = sorted(dic.values())
        # sort by values
        print("\nDict in ascending order:", asec)

        # Reverse the string
        desc = sorted(dic.values(), reverse=True)
        print("Dict in descending order:", desc)

    """ 
        2. Write a Python script to add a key to a dictionary. 
        Sample Dictionary : {0: 10, 1: 20}
        Expected Result : {0: 10, 1: 20, 2: 30}
    """

    # adding new key to previous dict using update()
    def addkey(self, dic):
        dic.update({"price": 100})
        print("Dict after adding element", dic)

    """3. Write a Python script to concatenate following dictionaries to create a new one. """

    # merging 3 dictionaries using update function
    def mergeing(self, dic1, dic2, dic3):
        dic1.update(dic2)
        dic1.update(dic3)
        return dic1

    # merging 3 dict using ** operator

    def merg(self, dic1, dic2, dic3):
        ress = {**dic1, **dic2, **dic3}
        return ress

    """ 4. Write a Python program to iterate over dictionaries using for loops. """

    # iterate over dictionary using for loop and show keys and values
    def iterdict(self, dic):
        print()
        # To loop over both key and value use dictionary.item()
        for ke, val in dic.items():
            print("key-> ", ke, "Value -> ", val)

    """	5. Write a Python script to generate and print a dictionary that contains a number (between 1 and n) 
        in the form (x, x*x). Sample Dictionary ( n = 5) : 
        Expected Output : {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
        """

    # Generate & print dic, dd is am empty dictionary, num input taken by user, to show result till num
    def showdic(self, dd, num):
        for temp in range(1, num + 1):
            dd[temp] = [temp * temp]
        print(dd)

    """ 6. Write a Python program to remove a key from a dictionary. """

    # Remove Key in given dictionary
    def delelement(self, newdic):
        # popitems remove last values in dic
        newdic.popitem()
        print(newdic)
        print("------------OR--------------------")
        # pop function remove specific element
        newdic.pop(1)
        print(newdic)
        print("------------OR--------------------")
        # del specific key
        if 3 in newdic:
            del newdic[3]
            print("\nDictionary after removing element: ", newdic)

    """ 7. Write a Python program to print all unique values in a dictionary. """

    # Check unique values in dict and print
    def uniqshow(self, dicti):
        # set(dicti)
        """Python dictionary is a mapping of unique keys to values.But
         Dictionaries are mutable, which means they can be changed.
         And in set every element is unique (no duplicates) and set is mutable.
        """
        # set does not support duplicate values
        u_value = set(val for dic in dicti for val in dic.values())
        print("\nUnique values in a dict: ", u_value)

    """ 
        8. Write a Python program to create a dictionary from a string. 
        Note: Track the count of the letters from the string. Sample string : 'w3resource'
        Expected output: {'3': 1, 's': 1, 'r': 2, 'u': 1, 'w': 1, 'c': 1, 'e': 2, 'o': 1}
        """

    # Count each letter of string in dictionary
    def strdict(self, string):
        # Create empty dict
        dictt = {}
        for val in string:
            # by default each values store as 0 for each key for new one add extra 1
            dictt[val] = dictt.get(val, 0) + 1
        print(dictt)

    """ 9. Write a Python program to print a dictionary in table format. """

    # show dict in table format
    def dicttable(self, my_dict):
        for row in zip(*([key] + value for key, value in sorted(my_dict.items()))):
            print(*row)

    # show dict in table format {:<10} {:<10} {:<15} this is a spacing format for colomn values
    def dicttable1(self, dicto):
        print(" {:<10} {:<10} {:<15}".format('colm1', 'colm2', 'colm3'))

        for k, v in dicto.items():
            # declaring variable for values in table
            colm2, colm3 = v
            print("{:<10} {:<15} {:<18}".format(k, colm2, colm3))

    """ 10. Write a Python program to count the values associated with key in a dictionary
        Sample data: = [{'id': 1, 'success': True, 'name': 'Lary'}, {'id': 2, 'success': False, 'name': 'Rabi'}, 
        {'id': 3, 'success': True, 'name': 'Alex'}]
        Expected result: Count of how many dictionaries have success as True

    """

    # Count values in dict for success = True
    def countdicval(self, sampledata):

        print("\nCount of success = True is: ", sum(dict2['success'] for dict2 in sampledata))

    """11. Write a Python program to convert a list into a nested dictionary of keys."""

    # convert a list into a nested dict
    def nesteddictlist(self, list12):
        dict12 = ttt = {}
        for temp in list12:
            ttt[temp] = {}
            ttt = ttt[temp]
        print("\nNested dictionary", dict12)

    # nested list into dictionary matrix format
    def newnested(self, data):
        new_result = {(temp1, temp2): data[temp2][temp1] for temp2 in range(len(data)) for temp1 in range(len(data[0]))}
        print("Nested list into dict. in matrix format: ", new_result)

    """ 12. Write a Python program to check multiple keys exists in a dictionary. """

    # Check for key existence
    def checkkeyexist(self, dicto):
        print("\nCheck key exist in dict or not: ")
        print(dicto.keys() >= {'amount', 'name'})
        print(dicto.keys() >= {'amount', 100})
        print(dicto.keys() >= {'name', 'Poonam'})
        print(dicto.keys() >= {'roll_id', 'class'})

    """ 13. Write a Python program to count number of items in a dictionary value that is a list. """

    # Check how many items in a dictionary value,  that is a list
    def countitems(self, dicto):
        # map return map object of all values(i.e len) in list and sum () convert or
        # count those values and show in readable format
        rr = sum(map(len, dicto.values()))
        print("\nCount of values in list:", rr)

    # print("---------------------------------------SET--------------------------------------------------")
    """1. Write a Python program to create a set. """

    # empty set
    def setcreation(self):
        seta = set()
        print("Empty set:", seta)

        setb = set([1, 2, 3, 4, 5])
        print("Set with list", setb)

        set1 = {'Red', 'Green', 'White', 'Black', 30, 'Yellow'}
        print("Created set: ", set1)
        print(type(set1))
        return setb

    def crestesetbyuser(self):
        global resultofset
        try:
            arrr = list()
            size = input("Enter the size of an Set:")
            # we have to typecast num to compare with length of string
            num2 = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter the elements: ")
                for ele in range(num2):
                    res = int(input())
                    arrr.append(res)
                    resultofset = set(arrr)
                # print("Set Elements:", set(arrr))
                return resultofset

            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    """2. Write a Python program to iteration over sets. """

    # set iteration using for loop
    def iterforset(self, set1):
        s = []
        for temp in set1:
            s.append(temp)
        return set(s)

    """3. Write a Python program to add member(s) in a set. """

    # Add new element in set
    def addinset(self, set1):
        set1.add("Colors")
        set1.update(["orange", "Blue", "Grey"])
        return set1

    """4. Write a Python program to remove item(s) from set 
    5. Write a Python program to remove an item from a set if it is present in the set. """

    # remove items in set using remove() and discard()
    def remitems(self, set1):
        try:
            set1.remove("orange")
            set1.discard("Grey")
            # If item will not present in set Discard() DOES NOT show error
            set1.discard("abcd")
            print("\nSet after removing items:", set1)
            # If item will not present in set remove() show error
            # set1.remove("abcd")
            return set1
        except KeyError:
            print("Element is not present in set")

    """6. Write a Python program to create an intersection of sets. """

    # Show intersection of two sets using intersection()
    def intersectionset(self, seta, setc):
        show = seta.intersection(setc)
        print("Intersection of two sets: ", show)

    """7. Write a Python program to create a union of sets. """

    # Show intersection of two sets using union()
    def unionset(self, setb, setc):
        show = setb.union(setc)
        print("\nUnion of two sets:", show)

    """8. Write a Python program to create set difference. """

    # Show intersection of two sets using difference()
    def setdiff(self, seta, setb):
        show = seta.difference(setb)
        print("Difference bet set1 over set2:", show)
        show1 = setb.difference(seta)
        print("Difference bet set2 over set1:", show1)

    """9. Write a Python program to create a symmetric difference. """

    # Show intersection of two sets using symmetric_difference()
    def setsymmetricdiff(self, seta, setb):
        show = seta.symmetric_difference(setb)
        print("Difference bet set1 over set2:", show)
        show1 = setb.symmetric_difference(seta)
        print("Difference bet set2 over set1:", show1)

    """10. Write a Python program to clear a set. """

    # Clear set
    def setclear(self, setz):
        print("\nOriginal set: ", setz)
        setz.clear()
        print("After clear set: ", setz)

    """11. Write a Python program to use of frozensets. """

    # Frozen set is just an immutable version of a Python set object.
    # While elements of a set can be modified at any time, elements of frozen set remains the same after creation.
    # this function shoe
    def usefrozenset(self):
        # It is empty frozenset
        fs = frozenset()
        print("\nEmpty Frozenset: ", fs)
        print(type(fs))
        # It is immutable
        # frozenset from iterable
        listvowel = ['a', 'e', 'i', 'o', 'u']
        fs1 = frozenset(listvowel)
        print("Frozon set: ", fs1)

        tuplenum = (1, 2, 3, 4, 2, 3, 5)
        fs3 = frozenset(tuplenum)
        print("Frozen set: ", fs3)

        Student = {"name": "Poonsm", "age": 21, "sex": "Female",
                   "college": "MCOE", "address": "Pune"}
        # making keys of dictionary as frozenset
        key1 = frozenset(Student)
        # printing keys details
        print('The frozen set is:', key1)

    """12. Write a Python program to find maximum and the minimum value in a set. """

    # Show min and max values
    def minmax(self, set1):
        try:
            min1 = min(set1)
            print("Minimum ele in set: ", min1)
            max1 = max(set1)
            print("Maximum ele in set: ", max1)
        except TypeError:
            print("Set is not valid")

    # print("---------------------------------------TUPLE--------------------------------------------------")
    """
        1. Write a Python program to create a tuple. 
        2. Write a Python program to create a tuple with different data types. 
        3. Write a Python program to unpack a tuple in several variables. 
    """

    def creattuple(self):
        # tuple with same datatype as string # packing tuple i.e. create tuple and assign to variable
        tup = ('apple', 'mango', 'pineapple', 'cherry', 'banana')
        print("\nCreated tuple: ", tup)

        # tuple with diff data types
        tup1 = (10, 'mango', 15.70, 'cherry', 'b')
        print("\nTuple with diff DataTypes: ", tup1)

        # unpacking tuple, It allows only for string
        (apple, mango, pineapple, cherry, banana) = tup
        print("\nUnpack Tuple", tup)

        tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7
        return tupl

    """4. Write a Python program to create the colon(:) of a tuple. """

    def clonetuple(self, tuplex):
        print("\nOriginal Tuple", tuplex)
        ss = tuplex[2:]
        print("From 2nd index using colon:", ss)
        # make a copy of a tuple using deepcopy() function
        # deepcopy doesn't affect on original tuple
        clont = copy.deepcopy(tuplex)
        # AttributeError if that positional element not empty

        # append operation perform only on list not on tuple
        clont[2].append(50)
        print("Copy tuple using deepcopy:", clont)
        print("After copy original tuple: ", tuplex)

    """5. Write a Python program to find the repeated items of a tuple. """

    def show(self, tupl):
        # count values from tuple
        store = Counter(tupl)
        print()
        list1 = []
        for ke, val in store.items():
            print("key: ", ke, "val: ", val)
            # check for repeated values
            if val > 1:
                list1.append(ke)
        return list1

    """6. Write a Python program to check whether an element exists within a tuple. """

    def eleexist(self, tup, num):
        if num in tup:
            print(True)
        else:
            print(False)

    """7. Write a Python program to convert a list to a tuple. """
    lis = [1, 2, 3, 4, 5]

    def conlis(self):
        print("Original List: ", self.lis)
        conv = tuple(self.lis)
        return conv

    # You can not remove items from a tuple, but you can delete tuple completely
    # It shoes UnboundLocalError

    """8. Write a Python program to remove an item from a tuple. """

    # tupl = 2, 4, 5, 6, 2, 3, 4, 4, 7

    def removeitem(self, tupl):
        try:
            del tupl
            # after deleting tuple it shows exception
            return tupl
        except (UnboundLocalError, NameError) as ex:
            print("Tuple is removed", ex)

    """9. Write a Python program to slice a tuple.  """

    # slice tuple
    def slicingtup(self, tupl):
        print("Start from 2nd index: ", tupl[2:])
        print("End at 1st index: ", tupl[:2])
        print("Tuple skipping one ele: ", tupl[::2])

    """10. Write a Python program to reverse a tuple. """

    def reversetup(self, tupl):
        resul = tupl[::-1]
        # aa = tuple(reversed(tupl)) # or we can use this also
        return resul

    # print("---------------------------------------LIST--------------------------------------------------")
    list1 = [10, 20, 30, 40]
    """
            @param lis
            Show addition of all elements in a list
    """
    # addition using loop
    def add(self):
        sum12 = 0
        for temp in self.list1:
            sum12 += temp
        print("\nOriginal List: ", self.list1)
        return sum12

    # using lambda
    def additn(self):
        print("\nOriginal List: ", self.list1)
        return reduce(lambda x, y: x + y, self.list1)

    def multiplication(self):
        print("\nOriginal List: ", self.list1)
        return reduce(lambda x, y: x * y, self.list1)

    """   3. Write a Python program to get the smallest number from a list.  """

    def min_max(self):
        print("\nOriginal List: ", self.list1)
        small = min(self.list1)
        print("Smaller element in a list: ", small)

        large = max(self.list1)
        print("Greatest element in a list: ", large)

    """ Write a Python program to count the number of strings where the string length is 2 or more 
        and the first and last character are same from a given list of strings. 
        Sample List : ['abc', 'xyz', 'aba', '1221'] Expected Result : 2
    """

    # for static methods we can pass argument from another class

    def countchar(self, list1):
        for temp in list1:
            length = len(temp)
            if length >= 2:
                if temp[0] == temp[length - 1]:
                    print("\nSize > than 2 and first & last char are same:", temp)

    list12 = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]

    """ 5. Write a Python program to get a list, sorted in increasing order by the last element in each 
    tuple from a given list of non-empty tuples.Sample List : [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
    Expected Result : [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)]
    """

    @staticmethod
    def last(lis):
        return lis[-1]

    def sortlist(self, list11):
        list12 = list(list11)
        return sorted(list12, key=self.last)

    # 6. Write a Python program to remove duplicates from a list.

    def withoutduplicate(self, samplelist):

        print("\nOriginal List: ", samplelist)
        result = list(dict.fromkeys(samplelist))
        print("list without duplicate elements: ", result)

    # 7. Write a Python program to clone or copy a list.
    def copylist(self, list1):
        l2 = list(list1)
        l2 = list1[:]
        print("\nOriginal list: ", l2)

        l3 = copy.copy(list1)
        print("Copy list: ", l3)

    # 8. Write a Python program to find the list of words that are longer than n from a given list of words.

    def strlen(self, list12, num):
        try:
            # we have to typecast num to compare with length of string
            num2 = int(num)
            if num.isdigit():
                for temp in list12:
                    leng = len(temp)
                    if leng >= num2:
                        print(temp)

                print("Not that size element is present")
            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    # 9. Write a Python function that takes two lists and returns True if they have at least one common member.
    slist1 = ['Red', 'Green', 'White', 'Red', 'Pink', 'Red']

    def usefilter(self, slist1):
        slist2 = ['Red', 'Wahhh', 'Ppp']

        if slist1 in slist2:
            return True
        else:
            return False

    """10. Write a Python program to print a specified list after removing the 0th, 4th and 5th elements. 
    Sample List : ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
    Expected Output : ['Green', 'White', 'Black']
    """

    def skipcolor(self, samplelist):
        for temp, temp2 in enumerate(samplelist):
            if temp not in (0, 4, 5):
                print(temp2)

    """ 11. Write a Python program to generate all permutations of a list in Python. """

    # # using built in functions
    def permu(self, listt):
        # Get all permutations of [1, 2, 3],size for permu
        perm = permutations(listt, 1)
        for temp in list(perm):
            print(temp)

        perm = permutations(listt, 3)
        # Print the obtained permutations
        for i in list(perm):
            print(i)

    """ 	12. Write a Python program to get the difference between the two lists. """

    def difference(self, list1, list2):
        zresult = set(list1) - set(list2)
        return zresult

    def all_perms(self, elements):
        if len(elements) <= 1:
            yield elements
        else:
            # If you swap the two for-loops, you get a sorted version
            for perm in self.all_perms(elements[1:]):
                for temp in range(len(elements)):
                    # nb elements[0:1] works in both string and list contexts
                    yield perm[:temp] + elements[0:1] + perm[temp:]

    """ 	13. Write a Python program to append a list to the second list.  """

    # using append
    # stack1 = ['a', 'b', 'c']
    # stack2 = ['d', 'e', 'f']
    def appendlist1(self, stack1, stack2):
        stack1.append(stack2)
        return stack1

    # Using Extend
    def appendlist23(self, list1, list2):
        list1.extend(list2)
        return list1

    """14. Write a python program to check whether two lists are circularly identical.  """
    """ Circulat identical"""

    def ciridenti(self, list1, list2):
        print(' '.join(map(str, list2)) in ' '.join(map(str, list1 * 2)))

    """15. Write a Python program to find common items from two lists. """

    def listcommonitem(self, list1, list2):
        print("Common item: ")
        for temp in list1:
            if temp in list2:
                print(temp)

    """16. Write a Python program to split a list based on first character of word.  """

    def spitlistbyword(self, exlist):
        for lett, words in groupby(sorted(exlist), key=itemgetter(0)):
            print(lett)
            for word in words:
                print(word)

    """ 17. Write a Python program to remove duplicates from a list of lists.  
        Sample list : [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
        New List : [[10, 20], [30, 56, 25], [33], [40]]"""

    def removeduplicate(self, slist):
        print("\nOriginal List: ", slist)
        slist.sort()
        a = list(slist for slist, _ in itertools.groupby(slist))

        print("list without duplicate elements: ", a)

    # print("--------------------------------String-----------------------------------------")
    """1. Write a Python program to calculate the length of a string."""

    def strlen(self):
        str1 = input("\nEnter the string: ")
        try:
            if str1.isalpha():
                length = len(str1)
                print("Length: ", length)
                return str1
            else:
                raise ValueError

        except ValueError:
            print("Not valid string")

    """
          2. Write a Python program to count the number of characters (character frequency) in a string.
          Sample String : google.com
          Expected Result : {'o': 3, 'g': 2, '.': 1, 'e': 1, 'l': 1, 'm': 1, 'c': 1}
      """

    def countcharinstr(self, str1):
        store = Counter(str1)
        print()
        for ke, val in store.items():
            print("key: ", ke, "val: ", val)

    """
           3. Write a Python program to get a string from a given string where 
           all occurrences of its first char have been changed to '$', except the first char itself.
           Sample String : 'restart'
           Expected Result : 'resta$t'
       """

    def strtodolor(self, str1):
        char1 = str1[0]
        str1 = str1.replace(char1, '$')
        str1 = char1 + str1[1:]
        return str1

    """
    4. Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
    If the given string already ends with 'ing' then add 'ly' instead. I
    f the string length of the given string is less than 3, leave it unchanged.
    Sample String : 'abc'
    Expected Result : 'abcing'
    Sample String : 'string'
    Expected Result : 'stringly'
    """

    def adding(self, str1):
        leng = len(str1)
        if leng >= 3:
            if (leng - 3) == 'ing':
                str12 = str1 + "ly"
                print("\nString : ", str12)
            else:
                str13 = str1 + "ly"
                print("\nString : ", str13)

    """5. Write a Python function that takes a list of words and returns the length of the longest one."""

    def takelist(self):
        try:
            strco = input("\nCount of strings:")
            strcount = int(strco)
            if strco.isdigit():
                print("Enter strings")
                llist = []
                for temp in range(0, strcount):
                    store = input(temp)
                    # two brack for 2 para vals
                    llist.append((len(store), store))
                    # sort in asending order
                    llist.sort()
                    # biggest will be last element
                print("Str with max length: ", llist[-1][1])
            else:
                raise ValueError
        except ValueError:
            print("Enter only int value")

    """6. Write a Python script that takes input from the user and displays that input back in upper and lower cases."""

    def upperlower(self, str5):
        upp = str5.upper()
        print("\nUpper case:", upp)
        low = str5.lower()
        print("Lower case:", low)

    """
    7. Write a Python program that accepts a comma separated sequen
    ce of words as input and prints
     the unique words in sorted form (alphanumerically).
    Sample Words : red, white, black, red, green, black
    Expected Result : black, green, red, white,red
    """

    def sortstrs(self):
        try:
            inp = input("\nEnter the size of list:")
            strcount = int(inp)
            if inp.isdigit():
                print("Enter strings")
                llist = []
                for temp in range(0, strcount):
                    store = input(temp)
                    llist.append(store)
                    llist.sort()
                return llist
            else:
                raise ValueError

        except Exception as ex:
            print("Valid Num plz", ex)

    """
    8. Write a Python program to get the last part of a string before a specified character.
    https://www.w3resource.com/python-exercises
    https://www.w3resource.com/python
    """

    def lastpartstr(self, strnew):
        print()
        # rsplit -> it search from end , and 2nd arg is
        print(strnew.rsplit('-', 1)[0])
        # rparition , it is faster and split only once
        print(strnew.rpartition('-')[0])

    """9. Write a Python program to display formatted text (width=50) as output."""

    def formatetxt(self, text):
        # Bydefault width is 70
        tex = textwrap.fill(text, width=50)
        return tex

    """10. Write a Python program to count occurrences of a substring in a string."""

    def countsubstr(self, strnew, strr):
        result1 = strnew.count(strr)
        return result1

    """11. Write a Python program to reverse a string."""

    def reverse11(self, string):
        string = string[::-1]
        return string

    """12. Write a Python program to lowercase first n characters in a string."""

    def func(self, str9):
        # strz= len(str9)
        print("1st Lower character: ", str9[:1].lower() + str9[1:])
