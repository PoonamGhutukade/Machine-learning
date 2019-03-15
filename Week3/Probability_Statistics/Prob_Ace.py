
from Week3.Utility.Util import UtilClass


class Probability:
    # class constructor
    def __init__(self):
        # private variable __card for Sample space
        self.__cards = 52
        # Util class object created to call method from that class
        self.obj = UtilClass()

    def calling(self):
        """Write a program to find probability of drawing an ace from pack of cards"""
        # outcomes
        ace = 4
        # round up value is up to 2 decimal
        print("\nProbability of ace: ", round(self.obj.aceprobability(ace, self.__cards), 2), '%')


# create class object to call its method
obj1 = Probability()
obj1.calling()
