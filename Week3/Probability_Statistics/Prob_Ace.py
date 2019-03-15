
from Week3.Utility.Util import UtilClass
import re


class Probability:
    # class constructor
    def __init__(self):
        # private variable for Sample space
        self.__cards = 52
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
