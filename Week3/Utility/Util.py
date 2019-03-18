from itertools import permutations
import itertools
import numpy as np


class UtilClass:
    """1. Write a python program to add below matrices """

    @staticmethod
    def matrix_addtn(matrix1, matrix2, result):
        # iterate throw row
        for temp in range(len(matrix1)):
            a = matrix1[temp]

            # iterate throw column
            for temp2 in range(len(matrix1[0])):
                result[temp][temp2] = a[temp] + matrix2[temp][temp2]
                result[temp][temp2] = matrix1[temp][temp2] + matrix2[temp][temp2]

        return result

    """4. Write a program to multiply matrices in problem 1"""

    # matrix multiplication using zip
    def matrix_multiplication(self, matrix1, matrix2):
        result = [[sum(row * col for row, col in zip(matrix1_row, matrix2_col))
                   for matrix2_col in zip(*matrix2)] for matrix1_row in matrix1]
        return result

    # matrix multiplication thought for loops
    def multiplication1(self, matrix1, matrix2, result14):
        # iterate by row of matrix1
        for temp in range(len(matrix1)):

            # iterating by coloum of matrix2
            for jtemp in range(len(matrix2[0])):

                # iterating by rows of matrix2
                for kval in range(len(matrix2)):
                    # calculate multiplication & store result
                    result14[temp][jtemp] += matrix1[temp][kval] * matrix2[kval][jtemp]
        return result14

    """2.Write a program to perform scalar multiplication of matrix and a number"""

    # scalar matrix multiplication method
    @staticmethod
    def matrix_scalar_multi(matrix1, result):
        # number multiply with each value into array
        num = 9
        # iterate throw row
        for temp in range(len(matrix1)):
            # iterate throw column
            for temp2 in range(len(matrix1[0])):
                # calculate multiplication & store result
                result[temp][temp2] = num * matrix1[temp][temp2]
        return result

    """3. Write a program to perform multiplication of given matrix and vector"""

    # multiplication of given matrix1 and vector
    def vectormultiplication(self, vector1, matrix1):
        # calculate length of vector and matrix
        row = len(vector1)
        mat = len(matrix1)
        # use zero matrix to store result into it
        multi = [0] * mat
        sum1 = 0
        # iterate throw row
        for temp in range(mat):
            row1 = matrix1[temp]

            # iterate throw column
            for temp2 in range(row):
                # calculate multiplication & store result
                sum1 += row1[temp2] * vector1[temp2]

            multi[temp] = sum1
            sum1 = 0
        return multi

    """5. Write a program to find inverse matrix of matrix X in problem 1"""

    def inverse_matrix(self, matrix1):
        return np.linalg.inv(matrix1)

    """	6. Write a program to find transpose matrix of matrix Y in problem 1 """

    # transpose matrix using list comprehension
    def transpose_matrix(self, matrix1):
        # use list comprehension
        rez = [[matrix1[j][i] for j in range(len(matrix1))] for i in range(len(matrix1[0]))]
        print("\n")
        for row in rez:
            print(row)

    # transpose matrix using zip
    def trans_matrix(self, matrix1):
        # using zip, show result into tuple
        t_matrix = zip(*matrix1)
        for row in t_matrix:
            print(row)

    # transpose matrix using numpy
    def transpose(self, matrix1):
        # use numpy inbuilt methods
        return np.transpose(matrix1)

    # ______________________________________________________________________________________________________
    """1. Write a program to find probability of drawing an ace from pack of cards"""

    def probability(self, outcomes, total_set):
        # probability = outcomes / total_set * 100
        return outcomes / total_set * 100

    """2. Write a program to find the probability of drawing an ace after drawing a king on the first draw """

    def ace_after_king(self, total_set):
        ace = 4
        # total there are 52 cards , 1 is drawn
        card_drawn = 1
        cards = total_set - card_drawn
        # prob of drawing an ace after drawing a king on the first draw
        # aceprob = self.probability(ace, cards)
        return self.probability(ace, cards)

    """3. Write a program to find the probability of drawing an ace after drawing an ace on the first draw """

    def ace_after_ace(self, total_set):
        aces = 4
        # total there are 4 ace cards , 1 is drawn
        card_drawn = 1
        ace = aces - card_drawn
        # prob of drawing an ace after drawing a king on the first draw
        # aceprob = self.probability(ace, total_set)
        return self.probability(ace, total_set)

    # _____________________________________________________________________________________________________________

    # prob for exactly 3 heads
    def three_heads(self, list11):
        whole_count = len(list11)
        user1 = str(input("Enter HHH to find out its prob:"))

        for temp in list11:
            # matched the content from user and for HHH
            if temp == user1 == 'HHH':
                # if it is matched then find out probability
                prob = 1 / whole_count
                return prob

    # for items in list find out all combination as a  permutation
    def permu(self, listt, num):
        # 1st parameter -Get all permutations of elements in a list,2nd parameter - size for permutation,
        # and use inbuilt function from itertools
        perm = permutations(listt, num)
        list112 = []
        for i in list(perm):
            # print(i)
            list112.append(i)
        return list112

    # take input from permutation and removed its duplicate elements
    def removeduplicate(self, slist):
        print("\nOriginal List using permutation: ", slist)
        # 1st sort the list
        slist.sort()
        # check each element in group and remove duplicate
        removed = list(slist for slist, _ in itertools.groupby(slist))
        # print("list without duplicate elements: ", a)
        return removed

    # show probability of given possibilities
    def atleast_onehead(self, list1):
        print("Final List after removing duplicates:", list1)
        whole_count = len(list1)
        count = 0
        listz = []

        # check for each item in list
        for item in list1:
            # check for each character in that item
            for char in item:
                # check for exactly one head in item
                if char == 'H':
                    # increment that count for each item if H is found
                    count += 1
            if count == 1:
                listz.append(item)
            count = 0
        print("exactly one heads list: ", listz)
        count_onehead = len(listz)
        print("Count of possibility: ", count_onehead)
        print("\nProbability of getting exactly one heads:", count_onehead / whole_count)
        # return listz

    # show probability of given possibilities
    def two_head(self, list1):
        print("Final List after removing duplicates:", list1)
        whole_count = len(list1)
        count = 0
        listz = []
        # first_condition at lest one head means except {TTT} all combination has 'H' so,
        a_prob = whole_count - 1
        first_condition = a_prob / whole_count
        print("prob of at least one head:", first_condition)
        # check for each item in list
        for item in list1:
            # check for each character in that item
            for char in item:
                # check for exactly one head in item
                if char == 'H':
                    # increment that count for each item if H is found
                    count += 1
            if count >= 2:
                listz.append(item)
            count = 0
        print("\nAt least two head list: ", listz)

        count_onehead = len(listz)
        print("Count of possibility: ", count_onehead)
        second_cond = count_onehead / whole_count
        print("Probability of getting least two head:", second_cond)

        print("Final Result :", round((second_cond / first_condition), 2))

    # _____________________________________________________________________________________________________
    # prob for not rainy, traffic and i am not late
    # Probability (notRainy . Traffic . NotLate)
    @staticmethod
    def not_rainy_not_late(not_rainy, not_rainy_traffic, not_rainy_traffic_not_late):
        return not_rainy * not_rainy_traffic * not_rainy_traffic_not_late

    # Probability ( I am Late today)
    def prob_late(self, rainy_traffic_late, rainy_not_traffic_late, not_rainy_traffic_late, not_rainy_not_traffic_late):
        return round(rainy_traffic_late + rainy_not_traffic_late + not_rainy_traffic_late + not_rainy_not_traffic_late,
                     2)

    #  Probability (Rainy day, and I arrived Late)
    def prob_late_rain(self, rainy_traffic_late, rainy_not_traffic_late):
        return round(rainy_traffic_late + rainy_not_traffic_late, 2)

    # _____________________________________________________________________________________________________________

    """find the probability that a woman has cancer if she has a positive mammogram result"""

    def prob_positive_cancer(self, breast_cancer, not_breast_cancer, positive_breast_cancer,
                             false_breast_cancer):

        # prob of B and A i.e. P(B intersection A)
        prob_b_and_a = positive_breast_cancer * breast_cancer
        prob_notb_and_a = false_breast_cancer * not_breast_cancer
        # return final probability
        return round(prob_b_and_a / (prob_b_and_a + prob_notb_and_a), 4)
