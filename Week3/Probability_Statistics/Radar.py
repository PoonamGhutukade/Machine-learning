"""11. A radar unit is used to measure speeds of cars on a motorway.
The speeds are normally distributed with a mean of 90 km/hr and a standard deviation of 10 km/hr.
Write a program to find the probability that a car picked at random is travelling at more than 100 km/hr? """


class Radar:
    def __init__(self):
        # default +ve z value, z(1) = 0.8413
        self.z_one = 0.8413
        # Mean and SD values are given
        self.mean = 90
        self.sd = 10

    def calculate_probability_radar(self):
        # P(X > 100), here required value X = 100, mean = 90, standard Deviation(S.D.) = 10
        x = 100

        # calculate z value 1st to find out probability
        z = (x - self.mean) / self.sd
        print("\nZ value: ", z)
        # area to the left of z
        if z <= 1:
            # total area for whole bell is 1
            total_area = 1
            # prob = total_area - area to the left of 1 i.e. (X >= 100) -> (Z >= 1) = (1 - 0.8413)
            prob = total_area - self.z_one
            print("P (X >= 100) is: ", prob)

            print("Prob that someone is travelling over 100km/hr: ", round(prob * 100, 2),"%")


obj = Radar()
obj.calculate_probability_radar()
