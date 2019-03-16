"""What is the probability that it's not raining and there is heavy traffic and I am not late?
What is the probability that I am late?
Given that I arrived late at work, what is the probability that it rained that day?
"""
import re
from Week3.Utility.Util import UtilClass


class AllValues:
    # constructor created to store all values
    def __init__(self):
        # rainy value 1/3
        self.rainy = 1 / 3
        # not rainy 1- 1/3 = 2/3
        self.not_rainy = 2 / 3

        # rainy and traffic = 1/2
        self.rainy_traffic = 1 / 2
        # rainy not traffic = 1- 1/2 = 1/2
        self.rainy_not_traffic = 1 / 2

        self.rainy_traffic_late = 1 / 12
        self.rainy_traffic_not_late = 1 / 12
        self.rainy_not_traffic_late = 1 / 24
        self.rainy_not_traffic_not_late = 1 / 8

        self.not_rainy_traffic = 1 / 4
        self.not_rainy_not_traffic = 3 / 4

        self.not_rainy_traffic_late = 1 / 24
        self.not_rainy_traffic_not_late = 3 / 4
        self.prob_not_rainy_traffic_not_late = 1/8
        self.not_rainy_not_traffic_late = 1 / 16
        self.not_rainy_not_traffic_not_late = 7 / 16


class ProbOfRainTrafficLate(AllValues):
    # class constructor created
    def __init__(self):
        # from super class, init method override
        super(ProbOfRainTrafficLate, self).__init__()
        self.obj1 = UtilClass()

    def calling1(self):

        while True:
            try:
                print()
                print("1.  probability that it's not raining and there is heavy traffic and I am not late ""\n"
                      "2. What is the probability that I am late"
                      "\n""3. yI arrived late at work, what is the probability that it rained that day ""\n"
                      "4. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():

                    if choice == 1:
                        # Probability (notRainy . Traffic . NotLate)
                        print("Probability of Not rainy, traffic, not late:", self.obj1.not_rainy_not_late(self.not_rainy,
                            self.not_rainy_traffic, self.not_rainy_traffic_not_late))

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        # Probability ( I am Late today)
                        print("Probability of I am late Today:", self.obj1.prob_late(self.rainy_traffic_late,
                          self.rainy_not_traffic_late,self.not_rainy_traffic_late, self.not_rainy_not_traffic_late))
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        # Probability (Rainy day, and I arrived Late)
                        print("Probability of I arrived late and rainy day:", self.obj1.prob_late_rain
                        (self.rainy_traffic_late, self.rainy_not_traffic_late))

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
                        break

                else:
                    raise ValueError

            except ValueError as e:
                print("\nInvalid Input", e)


obj = ProbOfRainTrafficLate()
obj.calling1()
