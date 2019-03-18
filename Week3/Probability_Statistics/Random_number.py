"""12.  Write a program to find the probability of getting a random number from the interval [2, 7]"""
import random


class Random:
    def __init__(self):
        # total count is 6 from 2 to 7
        self.total_count = 6
        self.possibility = 1

    def generateNum(self):
        # Integer from 2 to 7, end point included
        num = random.randint(2, 7)
        print("random number is ", num)
        obj.display(num)

    def display(self, num):
        print("probability of getting a random number", num, " =", self.possibility / self.total_count)


# class Object created to call its methods
obj = Random()
obj.generateNum()
