import re
from Week4.Utility.Util import UtilClass


class SizeofElements:
    # class constructor
    def __init__(self):
        # create obj of util class
        self.obj1 = UtilClass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Find the number of elements,length in byte and total Bytes of an array""\n"
                      "2. Operation on array :""\n"
                      "3. Compare two arrays using numpy""\n"
                      "4. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        # find the number of elements of an array
                        self.obj1.array_size_ele()

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        """find common values between two arrays
                        find the set difference of two arrays
                        Set exclusive-or will return the sorted, unique values """
                        self.obj1.operation_on_two_arrays()

                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        # compare two arrays using numpy
                        self.obj1.compare_two_array()

                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        # Exit from code
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
                print("\nInvalid Input", e)


# create obj of our class to call its methods
obj = SizeofElements()

obj.calling()
