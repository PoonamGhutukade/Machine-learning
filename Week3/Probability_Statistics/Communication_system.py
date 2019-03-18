"""8. In a communication system each data packet consists of 1000 bits. Due to the noise,
 each bit may be received in error with probability 0.1. It is assumed bit errors occur independently.
Find the probability that there are more than 120 errors in a certain data packet."""


class Packet:
    def __init__(self):
        self.g_p = 0.1
        self.g_n = 1000
        self.g_mean = self.g_n * self.g_p
        self.g_standard_deviation = (self.g_p * (1 - self.g_p)) ** 0.5
        self.g_x = 120

    def prob(self):
        z_score = (self.g_x - self.g_mean) / (self.g_standard_deviation * (self.g_n ** 0.5))
        return z_score


obj = Packet()
print("\nZ score value: ", obj.prob())
z = 1 - 0.9821
print("Probability = ", z)
