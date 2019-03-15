class Matrix:
    # Setter method
    def __set__(self, matrix_passed):
        self.matrix_original = matrix_passed

    # Getter method to variable
    def __get__(self):
        return self.matrix_original

    # Function to print matrix
    def print_Matrix(self, matrix_passed):
        print("Matrix: ")
        for item in matrix_passed:
            print(item)

    # calculate the co-factors of the elements of specific row and column
    def cofactors(self, row, col, matrix_passed):
        list1 = list()
        for row1 in range(0, len(matrix_passed[row])):
            for col1 in range(0, len(matrix_passed[row])):
                # eliminating the current row and column and entering rest to the list
                if row1 != row and col1 != col:
                    list1.append(matrix_passed[row1][col1])
        counter = 0
        # multiplying the cross elements that is first and last element in the list
        item = (list1[counter] * list1[-(counter + 1)])
        counter += 1
        # subtracting the multiplication of the rest 2 values from the previous one calculated
        item = item - (list1[counter] * list1[-(counter + 1)])
        return item

    # returns the transpose of any matrix
    def transpose_Matrix(self, matrix_passed):
        # print("Creating transpose for a matrix")
        # new matrix will be calculated by exchanging row into column
        transpose_adj_matrix = [[matrix_passed[row][col] for row in range(0, len(matrix_passed[0]))]
                                for col in range(0, len(matrix_passed))]
        return transpose_adj_matrix

    # calculating adjoint of given matrix
    def calculate_adjoint(self, matrix_passed):
        adjoint_matrix = []
        for row in range(0, len(matrix_passed[0])):
            list1 = list()
            for col in range(0, len(matrix_passed)):
                # adding each element returned by co factor method in list
                list1.append(self.cofactors(row, col, matrix_passed))
            # the list contains the co factors of each row so appends for each row
            adjoint_matrix.append(list1)
        temp = -1
        # multiply each elements by + - + signs respectively
        for row in range(0, matrix_passed.__len__()):
            for col in range(0, matrix_passed[row].__len__()):
                temp *= -1
                adjoint_matrix[row][col] *= temp

        return adjoint_matrix

    # calculate determinant of given matrix
    def calculate_determinant(self, matrix_passed):
        temp = -1
        item = 0
        """determinant = (B[0][0]*(B[1][1]*B[2][2]-B[1][2]*B[2][1]) - B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
          + B[0][2]*(B[1][0]*B[2][1] - B[1][1]*B[2][0]))"""
        for col in range(0, len(matrix_passed)):
            cofactor = self.cofactors(0, col, matrix_passed)
            """ matrix co-factor signs[ +  -  +
                                        -  +  -
                                        +  -  + ] 
                                                     
            """
            # as we need to add first subtract second and add third that is + - + hence multiplying with -1
            temp *= -1
            item += temp * (matrix_passed[0][col] * cofactor)

        return item

    # multiply 1/ determinant * matrix
    def matrix_multiply(self, matrix1, matrix2):
        matrix3 = [[0, 0, 0]] * 3
        # iterate by row of matrix1
        for row in range(0, matrix1[0].__len__()):
            # iterating by coloum of matrix2
            for col in range(0, len(matrix2)):
                # iterating by rows of matrix2
                for count in range(0, matrix2[0].__len__()):
                    matrix3[row][col] += matrix1[row][count] * matrix2[count][col]
        # print("The multiplication is")
        self.print_Matrix(matrix3)

    # main inverse method to find out inverse of given matrix
    def inverse_matrix(self):
        matrix1 = [[1, 7, 3],
                   [4, 3, 6],
                   [7, 2, 4]]

        self.print_Matrix(matrix1)
        # calculate adj of given matrix
        adjoint_matrix = self.calculate_adjoint(matrix1)
        # calculate transpose of that adj matrix
        adjoint_matrix = self.transpose_Matrix(adjoint_matrix)

        determinant = self.calculate_determinant(matrix1)
        print("\nDeterminant is ", determinant)

        inverse_matrix = [[adjoint_matrix[row][col]/determinant
                           for col in range(0, len(matrix1[row]))] for row in range(0, len(matrix1))]
        # self.print_Matrix(inverse_matrix)

        print(" \n Inverse Of matrix is : (1 /", determinant, ') * Adj Matrix')
        self.print_Matrix(adjoint_matrix)
        print("\nInverse matrix is:")
        self.matrix_multiply(matrix1, inverse_matrix)


# Matrix class obj created to call its methods
matrix_object = Matrix()
matrix_object.inverse_matrix()
