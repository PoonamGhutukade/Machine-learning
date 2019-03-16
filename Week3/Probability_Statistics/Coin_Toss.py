""" You toss a fair coin three times write a program to find following:
What is the probability of three heads, HHH?
What is the probability that you observe exactly one heads?
Given that you have observed at least one heads, what is the probability that you observe at least two heads?
"""
import re
from Week3.Utility.Util import UtilClass


class CoinToss:

    # class constructor
    def __init__(self):
        # Util class object created to call method from that class
        self.obj = UtilClass()

    def colling1(self):

        while True:
            try:
                print()
                print("1. What is the probability of three heads, HHH ""\n"
                      "2. What is the probability that you observe exactly one heads"
                      "\n""3. you have observed at least one heads, what is the probability that "
                      "you observe at least two heads ""\n"
                      "4. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():

                    if choice == 1:

                        list11 = ['HHH', 'TTT', 'HTH', 'HTT', 'THH', 'THT', 'HHT', 'TTH']
                        # calculate probability of three heads
                        result12 = self.obj.three_heads(list11)
                        result123 = str(result12)
                        if re.match(result123, 'None'):
                            print("Content not matched")
                        else:
                            print("Probability of HHH: ", result123)
                        print("_______________________________________________________________________________")

                    elif choice == 2:

                        num = int(input("\nEnter number how many times you want to toss the coin: "))

                        if num == 1:
                            list1 = ['H', 'T']
                            # calculate probability exactly one heads
                            res = self.obj.removeduplicate(self.obj.permu(list1, num))
                            self.obj.atleast_onehead(res)

                        elif num == 2:
                            ll1 = ['T', 'H', 'T', 'H']
                            result1 = self.obj.removeduplicate(self.obj.permu(ll1, num))
                            self.obj.atleast_onehead(result1)

                        elif num == 3:
                            ll1 = ['T', 'H', 'T', 'H', 'H', 'T']
                            # list11 = ['HHH', 'TTT', 'HTH', 'HTT', 'THH', 'THT', 'HHT', 'TTH']
                            res = self.obj.removeduplicate(self.obj.permu(ll1, num))
                            self.obj.atleast_onehead(res)
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        num = int(input("\nEnter number how many times you want to toss the coin: "))

                        if num == 1:
                            list1 = ['H', 'T']
                            res = self.obj.removeduplicate(self.obj.permu(list1, num))
                            self.obj.two_head(res)

                        elif num == 2:
                            ll1 = ['T', 'H', 'T', 'H']
                            result1 = self.obj.removeduplicate(self.obj.permu(ll1, num))
                            self.obj.two_head(result1)

                        elif num == 3:
                            ll1 = ['T', 'H', 'T', 'H', 'H', 'T']
                            # list11 = ['HHH', 'TTT', 'HTH', 'HTT', 'THH', 'THT', 'HHT', 'TTH']
                            res = self.obj.removeduplicate(self.obj.permu(ll1, num))
                            self.obj.two_head(res)

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


# class object created
obj1 = CoinToss()
# call class method through its object
obj1.colling1()
