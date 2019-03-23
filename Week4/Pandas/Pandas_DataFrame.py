import re
from Week4.Utility.Util import UtilClass


class PandaDataFrame:

    def __init__(self):
        # create obj for utility class
        self.obj1 = UtilClass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Create and display a DataFrame from a specified dictionary .""\n"
                      "2. Display a summary of the basic information ""\n"
                      "3. Get the first 3 rows of a given DataFrame""\n"
                      "4. Select the 'name' and 'score' columns""\n"
                      "5. Select 'name' and 'score' columns in rows 1, 3, 5, 6 from data frame""\n"
                      "6. Select the rows where the number of attempts in the examination is greater than 2""\n"
                      "7. Count the number of rows and columns ""\n"
                      "8. Select the rows where the score is missing, i.e. is NaN""\n"
                      "9. select the rows where attempts is less than 2 and score greater than 15""\n"
                      "10. Change the score in row 'd' to 11.5""\n"
                      "11. Calculate the sum of the examination attempts""\n"
                      "12. Calculate the mean score for each different student""\n"
                      "13. Append a new row 'k' to data frame""\n"
                      "14. Sort the DataFrame""\n"
                      "15. Replace the 'qualify' column contains the values 'yes' and 'no' with True and False""\n"
                      "16. Delete the 'attempts' column""\n"
                      "17. Insert a new column in existing DataFrame""\n"
                      "18. Iterate over rows in a DataFrame""\n"
                      "19. Get list from DataFrame column headers""\n"

                      "20. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        # pandas DF created and display
                        print("Pandas Data Frame:\n", self.obj1.create_panda_data_frame())
                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        # It will show the summary
                        print("Panda Data Frame Summary: ")
                        print(self.obj1.display_summary())
                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        rows_input = int(input("Enter the number to see rows"))
                        print(self.obj1.display_rows(rows_input))
                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        # select name and score from DF
                        print("Specific col\n ", self.obj1.show_specific_col())

                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        # specify column and rows from given DF
                        # name' and 'score' columns in rows 1, 3, 5, 6 from data frame
                        print("\n", self.obj1.display_row_from_col())
                        print("_______________________________________________________________________________")

                    elif choice == 6:
                        # select rows where no. of attempts greater than 2
                        # select specify row and column
                        print(self.obj1.show_greator_attemps())
                        print("_______________________________________________________________________________")

                    elif choice == 7:
                        # count number of rows and columns
                        self.obj1.count_rows_col()
                        print("_______________________________________________________________________________")

                    elif choice == 8:
                        # select rows where score is missing, i. e. NAN
                        print("\nRows where score is missing\n", self.obj1.null_score())
                        print("_______________________________________________________________________________")

                    elif choice == 9:
                        # Num of attempts in examination is 2 & score > 15
                        print("\nNum of attempts in examination is 2 & score > 15\n", self.obj1.num_attempts())
                        print("_______________________________________________________________________________")

                    elif choice == 10:
                        # Change the score in row 'd'
                        value = float(input("Enter new score for 'd' "))
                        print("Change the score in row 'd'\n", self.obj1.change_score(value))
                        print("_______________________________________________________________________________")
                    elif choice == 11:
                        # show sum of all attempts
                        print("Sum of attempts for each:\n", self.obj1.sum_attemps())
                        print("_______________________________________________________________________________")
                    elif choice == 12:
                        # calculate mean score
                        print("\nMean value for score: \n", self.obj1.calculate_mean())
                        print("_______________________________________________________________________________")

                    elif choice == 13:
                        # append new row k to data frame
                        self.obj1.add_del_new_row()
                        print("_______________________________________________________________________________")

                    elif choice == 14:
                        # sort data frame
                        print("Data frame after sorting\n")
                        print("Data after asc order sorting by score:\n",self.obj1.sorting())
                        print("_______________________________________________________________________________")

                    elif choice == 15:
                        # In quality column replace 'yes' and 'no' by 'true' and 'false'
                        print("\nData frame after replacing qualify values: \n", self.obj1.replace_values())
                        print("_______________________________________________________________________________")

                    elif choice == 16:
                        # delete attempt col from DF
                        print("DF after removing col attempts:\n", self.obj1.delete_dataframe_col())
                        print("_______________________________________________________________________________")

                    elif choice == 17:
                        # insert new column in existing DF
                        print("\nDF after inserting new col:\n ", self.obj1.inser_new_col())
                        print("_______________________________________________________________________________")

                    elif choice == 18:
                        # iterate over rows
                        self.obj1.iterate_rows()
                        print("_______________________________________________________________________________")

                    elif choice == 19:
                        # get list from DF col headers
                        print("Data frame column \n", self.obj1.get_cols())
                        print("_______________________________________________________________________________")

                    elif choice == 20:
                        exit()
                        print("_______________________________________________________________________________")

                    else:
                        print("Plz enter valid choice: ")

                    acc = str(input("IF you want to continue: type yes "))
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


# create class object to call its methods
obj = PandaDataFrame()
obj.calling()
