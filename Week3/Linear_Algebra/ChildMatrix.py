# from Week3.ParentMatrix import Parent
from Week3.Linear_Algebra.ParentMatrix import Parent


# single inheritance used here
# child class access constructor of parent class


class Child(Parent):

    # class parameterised Constructor
    def __init__(self, result):
        # from super class init method , we are accessing 2 matrices
        super(Child, self).__init__()
        self.result = result

    # matrix addition perform here
    def addition(self):
        self.result = [[self.matrix1[row][col] + self.matrix2[row][col] for col in range(len(self.matrix1[0]))] for row
                       in range(len(self.matrix1))]
        return self.result


result = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]
# class obj created
obj = Child(result)

# call matrix addition method and store result
result12 = obj.addition()
print("Matrix addition:")
# use loop to display matrix in proper format
for temp in result12:
    print(temp)
