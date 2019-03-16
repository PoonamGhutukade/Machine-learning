from Week3.Utility.Util import UtilClass
import re

class Probability:
    # class constructor
    def __init__(self):
        # private variable __card for Sample space
        self.__cards = 52

        # Util class object created to call method from that class
        self.obj = UtilClass()

    def calling(self):
        while True:
            try:
                print("1.  probability of drawing an ace from pack of cards ""\n"
                      "2. probability of drawing an ace after drawing a king on the first draw"
                      "\n""3. probability of drawing an ace after drawing an ace on the first draw ""\n"
                      "4 . Exit")

                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():

                    if choice == 1:
                        """1. Write a program to find probability of drawing an ace from pack of cards"""
                        # outcomes
                        ace = 4
                        # round up value is up to 2 decimal
                        print("\nProbability of ace: ", round(self.obj.probability(ace, self.__cards), 2), '%')

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        print("\nProb of ace after drawing a King", round(self.obj.ace_after_king(self.__cards), 2), '%')
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        print("\nProb of ace after drawing an Ace", round(self.obj.ace_after_ace(self.__cards), 2), '%')
                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        exit()

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


# create class object to call its method
obj1 = Probability()
# obj call its method
obj1.calling()
