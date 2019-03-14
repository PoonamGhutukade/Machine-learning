# dictionary is a collection which is unordered, unchangeable, & indexed, It doesn't repeat the value
from Utility.Util import UtilClass
import re


# create dict

class DictionaryP:
    # Utilityclass object is created to store and retrieve its values from that class
    obj = UtilClass()

    dic = {"fruit": "Mango",
           "Color": "Yellow",
           "test:": "Sweet"}

    while True:
        try:
            print()
            print("1. Sort ascending and descending a dictionary  ""\n""2. Add a key to a dictionary."
                  "\n""3. Concatenate dictionaries to create a new one ""\n"
                  "4. Iterate over dictionaries using for loops""\n"
                  "5. Generate and print a dictionary that contains a number (between 1 and n) ""\n"
                  "6. Remove a key from a dictionary""\n"
                  "7. Print all unique values in a dictionary.""\n"
                  "8. Count of the letters from the string""\n"
                  "9. Print a dictionary in table format""\n"
                  "10. Count the values associated with key in a dictionary. ""\n"
                  "11. Convert a list into a nested dictionary of keys""\n"
                  "12. Check multiple keys exists in a dictionary""\n"
                  "13. count number of items in a dictionary ""\n""14. Exit")
            ch = input("Enter choice:")
            choice = int(ch)
            if ch.isdigit():
                if choice == 1:
                    dic = {"fruit": "Mango",
                           "Color": "Yellow",
                           "test:": "Sweet"}
                    print("\nOriginal Dictionary: ", dic)
                    print("Keys: ", dic.keys())
                    print("Values: ", dic.values())
                    # show dic in asce and desc order by their values
                    obj.asecdec(dic)
                    print("_______________________________________________________________________________")

                elif choice == 2:
                    print("\nOriginal Dict: ", dic)
                    # adding new key to previous dict using update()
                    obj.addkey(dic)
                    print("_______________________________________________________________________________")

                elif choice == 3:
                    dic1 = {1: 10, 2: 20}
                    dic2 = {3: 30, 4: 40}
                    dic3 = {5: 50, 6: 60}
                    print("Original Dictionaries-> dict1:", dic1, ",dict2: ", dic2, ",dict3: ", dic3)
                    resl = obj.mergeing(dic1, dic2, dic3)
                    # merging 3 dictionaries using update function
                    print("Merging: ", resl)
                    print("---------------------------OR------------------------------------")
                    # merging 3 dict using ** operator
                    result = obj.merg(dic1, dic2, dic3)
                    print("Concatenating all dict: ", result)
                    print("_______________________________________________________________________________")

                elif choice == 4:
                    dic = {"fruit": "Mango",
                           "Color": "Yellow",
                           "test:": "Sweet"}
                    print("\nOriginal Dict: ", dic)
                    # iterate over dictionary using for loop and show keys and values
                    obj.iterdict(dic)
                    print("_______________________________________________________________________________")

                elif choice == 5:
                    # default empty dictionary
                    dd = dict()
                    num = int(input("\n\nEnter number for dic: "))
                    # Generate & print dic, dd is am empty dictionary, num input taken by user, to show result till num
                    obj.showdic(dd, num)
                    print("_______________________________________________________________________________")

                elif choice == 6:
                    newdic = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
                    print("\nOriginal Dict: ", newdic)
                    # Remove Key in given dictionary
                    obj.delelement(newdic)
                    print("_______________________________________________________________________________")

                elif choice == 7:
                    dicti = [{"V": "S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII": "S005"},
                             {"V": "S009"}, {"VIII": "S007"}]
                    print("\nOriginal Dict: ", dicti)
                    # Check unique values in dict and print
                    obj.uniqshow(dicti)
                    print("_______________________________________________________________________________")

                elif choice == 8:
                    string = 'PoonamGhutukade'
                    # Count each letter of string in dictionary
                    print("String is: ", string)
                    obj.strdict(string)
                    print("_______________________________________________________________________________")

                elif choice == 9:
                    my_dict = {'C1': [1, 2, 3], 'C2': [5, 6, 7], 'C3': [9, 10, 11]}
                    # show dict in table format
                    obj.dicttable(my_dict)
                    print("---------------------------OR------------------------------------")

                    dictionary = {1: ["Spices", 100],
                                  2: ["Other stuff", 50],
                                  3: ["Tea", 10],
                                  4: ["Contraband", 60],
                                  5: ["Fruit", 20],
                                  6: ["Textiles", 40]
                                  }

                    obj.dicttable1(dictionary)
                    print("_______________________________________________________________________________")

                elif choice == 10:
                    sampledata = [{'id': 1, 'success': True, 'name': 'Lary'},
                                  {'id': 2, 'success': False, 'name': 'Rabi'},
                                  {'id': 3, 'success': True, 'name': 'Alex'}]
                    print("\nOriginal Dict: ", sampledata)
                    # Count values in dict for success = True
                    obj.countdicval(sampledata)
                    print("_______________________________________________________________________________")

                elif choice == 11:
                    list12 = [1, 2, 3, 4, 5, 6]
                    print("\nOriginal List: ", list12)
                    # convert a list into a nested dict
                    obj.nesteddictlist(list12)

                    print("---------OR----------")
                    # take data as a in table cols and rows
                    data = [['1', '2', '-2'], ['3', '-1', '4']]
                    print("\nOriginal List: ", data)
                    # nested list into dictionary matrix format
                    obj.newnested(data)
                    print("_______________________________________________________________________________")

                elif choice == 12:
                    dictii = {
                        'name': 'Poonam',
                        'class': 'v',
                        'amount': 100,
                        'roll_id': '20'
                    }
                    print("\nOriginal Dict.: ", dictii)
                    # Check for key existence
                    obj.checkkeyexist(dictii)
                    print("_______________________________________________________________________________")

                elif choice == 13:
                    dicto = {'names': ['poonam', 'pooja', 'kranti', 'shweta'], 'surname': ['ghutukade']}
                    print("\nOriginal Dict.: ", dicto)
                    # Check how many items in a dictionary value,  that is a list
                    obj.countitems(dicto)
                    print("_______________________________________________________________________________")

                elif choice == 14:
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
