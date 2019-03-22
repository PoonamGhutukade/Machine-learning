"""10.  X is a normally normally distributed variable with mean μ = 30 and standard deviation σ = 4. Write a program to find
a. P(x < 40)
b. P(x > 21)
c. P(30 < x < 35)
"""
import re
from Week3.Probability_Statistics.Prob_Z_values import StoreValues


class NormalDistribution(StoreValues):
    def __init__(self):
        super(NormalDistribution, self).__init__()
        # Mean and SD values are given
        self.mean = 30
        self.sd = 4

    def calculate_probability(self):
        # P(X < 40), here required value X = 40, mean = 30, standard Deviation(S.D.) = 4
        x = 40

        # calculate z value 1st to find out probability
        z = (x - self.mean) / self.sd
        print("\nZ value: ", z)
        # area to the left of z
        if z <= 2.5:
            print("P (X < 40), Area to the left of ", z, "is: ", self.z_first)

    def calculate_probability1(self):
        # P(X > 21), here required value X = 21, mean = 30, standard Deviation(S.D.) = 4
        x1 = 21

        # calculate z value 1st to find out probability
        z1 = (x1 - self.mean) / self.sd
        print("\nZ value: ", z1)
        # area to the left of z
        if z1 >= -2.25:
            # total area for whole bell is 1
            total_area = 1
            # prob = total_area - area to the left of -2.25
            prob = total_area - self.z_second
            print("P (X > 21) is: ", prob)

    def calculate_probability2(self):
        # P(30 < X < 35), here required value X = 21, mean = 30, standard Deviation(S.D.) = 4
        x2 = 30
        x3 = 35

        # calculate z value 1st to find out probability
        z2 = (x2 - self.mean) / self.sd
        # It will give z value as zero
        print("\nFirst Z value: ", z2)

        z3 = (x3 - self.mean) / self.sd
        print("Second Z value: ", z3)

        # area to the left of z
        if z3 <= 1.25:

            # prob = area to the left of Z3 = 1.25 is 0.8944 and area to the left of Z2 = 0 is 0.5
            prob1 = self.z_third - 0.5
            print("P (30 < X < 35) is: ", prob1)

    def calling(self):
        while True:
            try:
                print()
                print("1. Probability  P(x < 40)""\n"
                      "2. P(x > 21)"
                      "\n""3. P(30 < x < 35) ""\n"
                      "4. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        obj.calculate_probability()
                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        obj.calculate_probability1()
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        obj.calculate_probability2()

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
                        break

                else:
                    raise ValueError

            except ValueError as e:
                print("\nInvalid Input", e)


# class Object created to call its methods
obj = NormalDistribution()
obj.calling()





