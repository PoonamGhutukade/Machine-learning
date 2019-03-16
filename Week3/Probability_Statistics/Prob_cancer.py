"""6. Given the following statistics, write a program to find the probability that a woman has cancer
if she has a positive mammogram result?
a. One percent of women over 50 have breast cancer.
b. Ninety percent of women who have breast cancer test positive on mammograms.
c. Eight percent of women will have false positives.
"""
from Week3.Utility.Util import UtilClass


class Givenvalues:
    def __init__(self):
        # 1 % womans have breaste cancer i.e P(A) = 0.01
        self.breast_cancer = 0.01
        # womans who doesnt have breast cancer i.e  1 - breast_cancer or (1- P(A))
        self.not_breast_cancer = 0.99

        #  Ninety percent of women who have breast cancer test positive i.e. P(B) = 0.9
        self.positive_breast_cancer = 0.9

        # Eight percent of women will have false positives i.e p(-B) = 0.008
        self.false_breast_cancer = 0.08

        self.obj1 = UtilClass()

    def colling(self):
        print("Probability of positive cancer :", self.obj1.prob_positive_cancer(self.breast_cancer,
                self.not_breast_cancer, self.positive_breast_cancer, self.false_breast_cancer))


# class object created to call its method
obj = Givenvalues()
obj.colling()
