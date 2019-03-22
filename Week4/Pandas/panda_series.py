import pandas as pd
import re
from Week4.Utility.Util import UtilClass


class PandaSeries:
    def __init__(self):
        self.obj1 = UtilClass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Create and display a one-dimensional array using Pandas module.""\n"
                      "2. Convert a Panda module Series to Python list ""\n"
                      "3. Add, subtract, multiple and divide two Pandas Series""\n"
                      "4. Get the powers of an array values element-wise""\n"

                      "5. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        size = input("Enter the size for list:")
                        panada1d_series = self.obj1.create_series(size)
                        print("One-dimensional array using Pandas Series:\n", panada1d_series)

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        size = input("Enter the size for list:")
                        panada1d_series1 = self.obj1.create_series(size)
                        string = str(panada1d_series1)
                        if re.match(string, 'None'):
                            # If set is empty create again and then show its iteration again
                            print("Empty series")
                            continue

                        else:
                            print("Pandas Series:\n", panada1d_series1)
                            print("Panda series to list: ", self.obj1.conversion(panada1d_series1))
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        series1 = pd.Series([2, 4, 6, 8, 10])
                        series2 = pd.Series([1, 3, 5, 7, 9])
                        self.obj1.series_operations(series1, series2)

                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        panda_series = pd.Series([1, 2, 3, 4])
                        print("Original panda series :", panda_series)
                        print("Power of 2 for all elements: ", self.obj1.series_power(panda_series))

                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        exit()
                        print("_______________________________________________________________________________")

                    else:
                        print("Plz enter valid choice: ")

                    acc = str(input("IF you want to continue: type yes "))
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


# create class object to call its methods
obj = PandaSeries()
obj.calling()
