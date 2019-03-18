"""7. A bank teller serves customers standing in the queue one by one.
Suppose that the service time XiXi for customer ii has mean EXi=2 (minutes) and Var(Xi)=1.
We assume that service times for different bank customers are independent.
Let YY be the total time the bank teller spends serving 50 customers. Write a program to find P(90<Y<110)
"""


class Bank:
    def __init__(self):
        self.mean = 2
        self.sd = 1 ** 0.5
        # total_customer
        self.value_n = 50
        # P(90<Y<110)
        self.value_1 = 90
        self.value_2 = 110

    def prob(self):
        # z_score1 = (l_x1 - l_mean * l_n) / (l_standard_deviation * (l_n ** 0.5))
        z_score1 = (self.value_1 - self.mean * self.value_n) / (self.sd * (self.value_n ** 0.5))
        z_score2 = (self.value_2 - self.mean * self.value_n) / (self.sd * (self.value_n ** 0.5))
        return z_score1,z_score2


obj = Bank()
print("Probability:", obj.prob())
z = 0.9207 - 0.793
# z values for -1.414213562373095 - 1.414213562373095
print("Final Z value: ", z)
