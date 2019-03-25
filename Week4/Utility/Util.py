import numpy as np
import numpy.matlib
import pandas as pd
import math
import matplotlib.pyplot as plt
# following for custom grid values
import datetime as DT
from matplotlib.dates import date2num
from keyrings import alt
from pylab import randn
import seaborn as sb
# import this for hover effect
import mplcursors


class UtilClass:
    # class constructor
    def __init__(self):
        # create pandas Data Frame
        self.exam_data = {
            'name': ['Poonam', 'Neha', 'Shweta', 'Harshad', 'Akanksha', 'Megha', 'John', 'Vidya', 'Roja', 'Dima'],
            'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
            'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
            'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        self.data_frame = pd.DataFrame(self.exam_data, index=self.labels)

    # _________________________________________________________________________________________________________________

    def convert_list_ndarray(self, original_list):
        # asarray is useful to converting python sequence into ndarray
        return np.asarray(original_list)

    # create new matrix
    def matrix_creation(self, input1, input2):
        try:
            # input1 = input("\nEnter the matrix start value:")
            # input2 = input("Enter the matrix end value:")
            start = int(input1)
            end = int(input2)
            if input1.isdigit() and input2.isdigit():
                # It display number from start to end input value
                return np.arange(start, end + 1)
            else:
                raise ValueError
        except Exception as e:
            print("\nInvalid Input:", e)

    # reshape created matrix
    def reshape_matrix(self, array1, num1, num2):
        try:
            first_dimension = int(num1)
            second_dimension = int(num2)
            if num1.isdigit() and num2.isdigit():
                # reshape array
                final_aaray = array1.reshape(first_dimension, second_dimension)
                # check datatype of given array, by default datatype is int64 for integer
                print("Array DT: ", np.array(final_aaray).dtype)
                # this return the number of array dimension
                print("\nArray Dimension for following array is :", final_aaray.ndim)

                return final_aaray
            else:
                raise ValueError
        except Exception as es:
            print("cannot reshape array :", es)

    def null_vector_creation(self, array2):
        # np.zeroes allow us to create array with zero fields
        array3 = (len(array2))
        # use nympy.matlib library for matrix operations
        # OR array4 = np.matrix.zeros(array3)
        return np.matlib.zeros(array3)

    def update_matrix(self, array4):
        # print("\nOriginal null vector array of size 10:", array4)
        # to update array value, syntax -> numpy.put(array, index, value, mode='raise')
        np.put(array4, 6, 11)
        return array4

    # reverse matrix method
    def matrix_reverse(self, array_created):
        return array_created[::-1]

    def matrix_one_creation(self, array2):
        # np.zeroes allow us to create array with zero fields
        array3 = (len(array2))
        return np.matlib.ones(array3)

    """find the number of elements of an array, length of one array element in bytes and
          total bytes consumed by the elements"""

    def array_size_ele(self):
        # OR x = np.array([1,2,3], dtype=np.float64)
        """
        for int8 -> one element = 1 byte('i1')
        for int16 -> one element = 2 bytes('i2')
        for int32 -> one element = 4 bytes('i4')
        for int64 -> one element = 8 bytes('i8')
        """
        # OR x = np.array([1, 2, 3, 4, 5], dtype = np.int64)
        x = np.array([1, 2, 3, 4, 5], np.dtype('i8'))
        # find the number of elements of an array
        print("Size of the array: ", x.size)
        # length of one array element in bytes
        print("Length of one array element in bytes: ", x.itemsize)
        #  bytes consumed by the elements
        print("Total bytes consumed by the elements of the array: ", x.nbytes)

    """find common values between two arrays
        find the set difference of two arrays
        Set exclusive-or will return the sorted, unique values """

    def operation_on_two_arrays(self):
        array1 = [0, 10, 20, 40, 60, 80]
        array2 = [10, 30, 40, 50, 70, 90]
        print("\nOriginal arr1: ", array1)
        print("Original arr2: ", array2)

        common = np.intersect1d(array1, array2)
        print("\nCommon from both lists: ", common)
        diff = np.setdiff1d(array1, array2)
        print("\nDiffrence  from both lists: ", diff)
        diff = np.setxor1d(array1, array2)
        print("\nUnique values that are in only one from both lists: ", diff)
        diff = np.union1d(array1, array2)
        print("\nUnion from both lists: ", diff)

    def compare_two_array(self):
        arr1 = [1, 2]
        arr2 = [3, 4]
        print("array1: ", arr1)
        print("array2: ", arr2)
        # check 2 arrays for less or greater condition
        print("arr1 > arr2: ", np.greater(arr1, arr2))
        print("arr1 >= arr2: ", np.greater_equal(arr1, arr2))
        print("arr1 < arr2 ", np.less(arr1, arr2))
        print("arr1 <= arr2: ", np.less_equal(arr1, arr2))

    def flattendarr(self, arr1):
        arr = np.ndarray.flatten(arr1)
        print("Dimension for below array :", arr.ndim)
        # convert nD array to 1D array
        return np.ndarray.flatten(arr1)

    """
            for float8 -> one element = 1 byte('f1')
            for float16 -> one element = 2 bytes('f2')
            for float32 -> one element = 4 bytes('f4')
            for float64 -> one element = 8 bytes('f8')
    """

    def change_dt(self, arr12):
        # change datatype int to float64
        return np.array(arr12, dtype='f8')

    def three_d_arr(self):
        output = self.reshape_matrix(self.matrix_creation())
        arr_new = np.array(output, ndmin=3)
        # print("3D Array: \n", arr_new)
        return arr_new

    def identitymatrix(self, arr):
        arr1 = len(arr)
        # np.matlib.eye only take integer number for array, so have to find out length of given array
        identity_matrix = np.matlib.eye(arr1, dtype=float)
        final_matrix = np.array(identity_matrix, ndmin=3)
        print("Dimension for below matrix: ", final_matrix.ndim)
        return final_matrix

    def concreate_data(self, data1, data2):
        # concatenate two array
        return np.concatenate((data1, data2), 1)

    # array multiply with each element by 3
    @staticmethod
    def matrix_scalar_multi(array, num):
        for temp in np.nditer(array, op_flags=['readwrite']): temp[...] = num * temp
        return array

    # __________________________________________________________________________________________________________________
    # pandas functions

    # create panda series
    def create_series(self, size):
        try:
            arrr = list()
            # we have to typecast num to compare with length of string
            num2 = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter the elements: ")
                for ele in range(num2):
                    res = int(input())
                    arrr.append(res)
                    # put list into panda series
                panda_series = pd.Series(arrr)
                return panda_series

            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    # conver panda_series to list
    def conversion(self, panda_series1):
        print(type(panda_series1.tolist()))
        return panda_series1.tolist()

    #
    def series_operations(self, series1, series2):
        add = series1 + series2
        print("\nAddition of two series\n", add)
        sub = series1 - series2
        print("Substraction of two series\n", sub)
        mul = series1 * series2
        print("Multiplication of two series\n", mul)
        div = series1 / series2
        print("Division of two series\n", div)

    def series_power(self, panda_series):
        result = []
        for temp in panda_series:
            res = math.pow(temp, 2)
            result.append(res)
        return result

    def create_panda_data_frame(self):
        # pandas DF created and display
        return self.data_frame

    def display_summary(self):
        # show the summary
        return self.data_frame.info()

    def display_rows(self, num):
        # get the 1st 3 rows of a given data frame
        print("\n First ", num, " rows of given data frame")
        return self.data_frame.iloc[:num]

    def show_specific_col(self):
        # select name and score from DF
        return self.data_frame[['name', 'score']]

    def display_row_from_col(self):
        # specify column and rows from given DF
        # name' and 'score' columns in rows 1, 3, 5, 6 from data frame
        return self.data_frame.iloc[[1, 3, 5, 6], [0, 1]]

    def show_greator_attemps(self):
        # select rows where no. of attempts greater than 2
        # select specify row and column
        return self.data_frame.ix[[1, 3, 5], ['name', 'score']]

    def count_rows_col(self):
        # count number of rows and columns
        rows_count = len(self.data_frame.axes[0])
        col_count = len(self.data_frame.axes[1])
        print("\nTotal rows in DF:", rows_count)
        print("Total col in DF:", col_count)

    def null_score(self):
        # select rows where score is missing, i. e. NAN
        return self.data_frame[self.data_frame['score'].isnull()]

    def num_attempts(self):
        # Num of attempts in examination is 2 & score > 15
        return self.data_frame[(self.data_frame['attempts'] < 3) & (self.data_frame['score'] > 15)]

    def change_score(self, score_input):
        val = float(score_input)
        # Change the score in row 'd'
        self.data_frame.loc['d', 'score'] = score_input
        return self.data_frame

    def sum_attemps(self):
        # show sum of all attempts
        print("Sum of attempts for each:\n", [self.data_frame['attempts'].sum])

    def calculate_mean(self):
        # calculate mean score
        return self.data_frame['score'].mean()

    def add_del_new_row(self):
        print("\nOriginal Data Frame:\n", self.data_frame)
        self.data_frame.loc['k'] = ['kranti', 15.5, 1, 'yes']
        print("\nDF after append new row:\n", self.data_frame)
        # it should store in another variable after deletion, otherwise it wont affect on previous DF
        self.data_frame = self.data_frame.drop('k')
        print("\nFinal DF after removing col \n: ", self.data_frame)

    def sorting(self):
        df_desc = self.data_frame.sort_values(by=['name'], ascending=[False])
        print("Sort data by name - desc order \n", df_desc)
        df_asc = self.data_frame.sort_values(by=['score'], ascending=[True])
        #  print("values".format(self.data_frame.sort_values(by=['name', 'score'], ascending=[False, True])))
        return df_asc

    def replace_values(self):
        # In quality column replace 'yes' and 'no' by 'true' and 'false'
        self.data_frame['qualify'] = self.data_frame['qualify'].map({'yes': 'True', 'no': 'False'})
        return self.data_frame

    def delete_dataframe_col(self):
        # delete attempt col from DF
        self.data_frame.pop('attempts')
        return self.data_frame

    def inser_new_col(self):
        # insert new column in existing DF
        print("\n Original DF:\n", self.data_frame)
        color = ['red', 'blue', 'black', 'pink', 'orange', 'yellow', 'grey', 'white', 'magenta',
                 'green']
        self.data_frame['color'] = color
        return self.data_frame

    def iterate_rows(self):
        # iterate over rows
        print("Iterate over rows: ")
        for index, rows in self.data_frame.iterrows():
            print(rows['name'], rows['score'])

    def get_cols(self):
        # get list from DF col headers
        return list(self.data_frame.columns.values)

    # ______________________________________________________MATPLOTLIB_and_PLOTLY_____________________________________________________________

    # draw new line
    def draw_line_matplotlib(self, x_axis, y_axis):
        print("X axis values", x_axis)
        print("Y axis values", y_axis)
        plt.plot(x_axis, y_axis)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title("Draw A line")
        plt.show()

        # create list

    def creatlist(self, size):
        try:
            empty_list = []
            res = 0
            # we have to typecast num to compare with length of string
            axis_size = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter values: ")
                for ele in range(axis_size):
                    res = int(input())
                    empty_list.append(res)
                return empty_list

            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    def read_text_file(self):
        # opens a file
        with open("test.txt") as f:
            # reads a file
            data = f.read()
        # split a data with new line
        data = data.split('\n')

        x = [row.split(' ')[0] for row in data]
        y = [row.split(' ')[0] for row in data]

        # plotting the line points
        plt.plot(x, y)

        # Set the x axis label of the current axis.
        plt.xlabel('x - axis')

        # Set the y axis label of the current axis.
        plt.ylabel('y - axis')

        # Set a title
        plt.title('Simple line, data from Text File')

        # Display a figure.
        plt.show()

    def read_data(self):

        # reading csv file
        df = pd.read_csv('test.csv', sep=',', parse_dates=True, index_col=0)
        # plotting lines
        df.plot()
        # Show the figure.
        plt.show()

    def custom_grid(self):
        data = [(DT.datetime.strptime('2016-10-03', "%Y-%m-%d"), 772.559998),
                (DT.datetime.strptime('2016-10-04', "%Y-%m-%d"), 776.429993),
                (DT.datetime.strptime('2016-10-05', "%Y-%m-%d"), 776.469971),
                (DT.datetime.strptime('2016-10-06', "%Y-%m-%d"), 776.859985),
                (DT.datetime.strptime('2016-10-07', "%Y-%m-%d"), 775.080017)]

        # Convert datetime objects to Matplotlib dates.
        x = [date2num(date) for (date, value) in data]
        y = [value for (date, value) in data]

        # creates object of figure class
        fig = plt.figure()

        graph = fig.add_subplot(1, 1, 1)

        # Plot the data as a red line with round markers
        graph.plot(x, y, 'r-o')

        # Set the xtick locations
        graph.set_xticks(x)

        # Set the xtick labels(date)
        graph.set_xticklabels([date.strftime("%Y-%m-%d") for (date, value) in data])

        # Set the x axis label
        plt.xlabel('Date')

        # Set the y axis label
        plt.ylabel('Closing Value')

        # Sets a title
        plt.title('Closing stock value of Alphabet Inc.')

        # Customize the grid
        plt.grid(linestyle='-', linewidth='0.5', color='blue')

        # shows the plot
        plt.show()

    def draw__multiple_line(self, x1_axis, y1_axis, x2_axis, y2_axis):

        # # plotting the line 1 points
        plt.plot(x1_axis, y1_axis, label="line 1")

        # plotting the line 2 points
        plt.plot(x2_axis, y2_axis, label="line 2")

        # Set the x axis label
        plt.xlabel('x - axis')
        # Set the y axis label
        plt.ylabel('y - axis')

        # Sets a title
        plt.title('Two or more lines on same plot with suitable legends, and color ')

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        # show a legend on the plot, on top right corner as line1 and line2
        plt.legend()

        # Display a figure.
        plt.show()

    def draw__line_with_color(self, x1_axis, y1_axis, x2_axis, y2_axis):

        # plotting the line 1 points with color,width,label
        plt.plot(x1_axis, y1_axis, color='blue', linewidth=2, label="line1-width-2")

        # plotting the line 2 points with color,width,label
        plt.plot(x2_axis, y2_axis, color='black', linewidth=4, label="line2-width-4")

        plt.title('Multiple lines on same plot with suitable legends , colors & width ')

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()

    def plot_line_with_style(self, x1_axis, y1_axis, x2_axis, y2_axis):
        # plotting the line 2 points with Dotted line style
        plt.plot(x1_axis, y1_axis, linestyle=':', label="line 2")

        # plotting the line 3 points with Dash-dot line style
        plt.plot(x2_axis, y2_axis, linestyle='-.', label="line 2")

        # Set the x & y axis label
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')

        # Sets a title
        plt.title('Two or more lines on same plot with diff Style ')

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()

    def line_marker(self, x1_axis, y1_axis, x2_axis, y2_axis):
        # plotting the line 1 points with marker
        plt.plot(x1_axis, y1_axis, label="line 1", marker='o', markerfacecolor='blue', markersize=12)

        # plotting the line 2 points
        plt.plot(x2_axis, y2_axis, label="line 2")

        # Set the x & y axis label
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')

        # Sets a title
        plt.title('Two or more lines on same plot with line marker')

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()

    def type_chart(self):
        x = range(10)

        y = range(10)

        plt.subplot(2, 2, 1)
        plt.plot(x, y)

        plt.subplot(2, 2, 2)
        plt.bar(x, y)

        plt.subplot(2, 2, 3)
        plt.hist(x, y)

        plt.subplot(2, 2, 4)

        plt.scatter(x, y)

        plt.show()

    def set_axis_values(self, x_axis, y_axis):

        # plotting the line 1 points
        plt.plot(x_axis, y_axis, label="line 1")

        # Set the x & y axis label
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')

        # Sets a title
        plt.title('Two lines on same plot and set new axis ')

        # returns current axis values
        print(plt.axis())
        # xmin,xmax,ymin,ymax=plt.axis()

        # accepting values to set new axis values
        print("set new axis limit")
        xmin = int(input("xmin val"))
        xmax = int(input("xmax val"))
        ymin = int(input("ymin val"))
        ymax = int(input("ymax val"))

        # sets new axis values
        plt.axis([xmin, xmax, ymin, ymax])

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()

    def plot_quantities(self, x1_axis, y1_axis, x2_axis, y2_axis):
        # Set the x & y axis label
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')

        # Sets a title
        plt.title(' quantities which have an x and y position ')

        # set new axes limits
        plt.axis([0, 100, 0, 100])

        # use pylab to plot x and y as red circles
        plt.plot(x1_axis, y1_axis, 'b*', x2_axis, y2_axis, 'ro')

        # shows the plot
        plt.show()

    def draw_line_with_diff_formats_witharray(self):
        # Sampled time at 200ms intervals
        t = np.arange(0., 5., 0.2)

        # green dashes, blue squares and red triangles
        plt.plot(t, t, 'g--', t, t ** 2, 'bs', t, t ** 3, 'r^')
        plt.show()

    def large_small_grid(self):
        data = [(DT.datetime.strptime('2016-10-03', "%Y-%m-%d"), 772.559998),
                (DT.datetime.strptime('2016-10-04', "%Y-%m-%d"), 776.429993),
                (DT.datetime.strptime('2016-10-05', "%Y-%m-%d"), 776.469971),
                (DT.datetime.strptime('2016-10-06', "%Y-%m-%d"), 776.859985),
                (DT.datetime.strptime('2016-10-07', "%Y-%m-%d"), 775.080017)]

        # Convert datetime objects to Matplotlib dates.
        x = [date2num(date) for (date, value) in data]
        y = [value for (date, value) in data]

        # creates object of figure class
        fig = plt.figure()

        graph = fig.add_subplot(1, 1, 1)

        # Plot the data as a red line with round markers
        graph.plot(x, y, 'r-o')

        # Set the xtick locations
        graph.set_xticks(x)

        # Set the xtick labels
        graph.set_xticklabels(
            [date.strftime("%Y-%m-%d") for (date, value) in data]
        )

        # Set the x axis label
        plt.xlabel('Date')
        # Set the y axis label
        plt.ylabel('Closing Value')
        # Sets a title
        plt.title('Closing stock value of Alphabet Inc.')

        # Turn on the minor TICKS, which are required for the minor GRID
        plt.minorticks_on()

        # Customize the major grid
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        # Turn off the display of all ticks.
        plt.tick_params(which='both',  # Options for both major and minor ticks
                        top='off',  # turn off top ticks
                        left='off',  # turn off left ticks
                        right='off',  # turn off right ticks
                        bottom='off')  # turn off bottom ticks

        # shows the plot
        plt.show()

    def create_subplot(self, x1_axis, y1_axis, x2_axis, y2_axis):

        # Sets up a subplot grid that has height 2 and width 1,
        # and set the first such subplot as active.
        plt.subplot(2, 1, 1)

        # plotting the line 1 points
        plt.plot(x1_axis, y1_axis, label="line 1")

        # Sets a title
        plt.title('subplot1')

        print("line 2")

        # Set the second subplot as active, and make the second plot.
        plt.subplot(2, 1, 2)

        # plotting the line 2 points
        plt.plot(x2_axis, y2_axis, label="line 2")

        # Sets a title
        plt.title('subplot2')

        # Shows the figure.
        plt.show()

    # _________________________________________________BARCHART_________________________________________________________________

    def draw_bar_vertically(self, language, popularity):

        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x and y axis values to create bar chart
        plt.bar(x_pos, popularity, color='blue')

        # set the current tick locations and labels(language name)of the x-axis
        plt.xticks(x_pos, language)
        plt.title("Vertical Bar")
        # Shows the figure.
        plt.show()

    def draw_bar_horizontally(self, language, popularity):
        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x and y axis values to create bar chart horizontally
        plt.barh(x_pos, popularity, color='green')

        # set the current tick locations and labels(language name) to y-axis
        plt.yticks(x_pos, language)
        plt.title("Horizontal Bar")
        plt.show()

    def draw_bar_with_uniform_color(self, language, popularity):
        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x and y axis values to create bar chart
        plt.bar(x_pos, popularity, color=(0.2, 0.4, 0.6, 0.6))

        # set the current tick locations and labels(language name)of the x-axis
        plt.xticks(x_pos, language)
        plt.title("Bar Chart with Uniform Color")
        # Shows the figure.
        plt.show()

    def draw_bar_with_diff_color(self, language, popularity):
        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x and y axis values to create bar chart
        plt.bar(x_pos, popularity, color=['black', 'red', 'green', 'blue', 'cyan'])

        # set the current tick locations and labels(language name)of the x-axis
        plt.xticks(x_pos, language)
        plt.title("Bar Chart with different Color")
        # Shows the figure.
        plt.show()

    def attach_label(self, language, popularity):

        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x and y axis values to create bar chart
        # plt.bar(x_pos, self.popularity, color='blue')

        # set the current tick locations and labels(language name)of the x-axis
        # plt.xticks(x_pos, self.language)

        fig, ax = plt.subplots()
        rects1 = ax.bar(x_pos, popularity, color='b')
        plt.xticks(x_pos, language)

        # for i, v in enumerate(rects1):
        #     ax.text(v + 3, i + .25, str(v), color='red', fontweight='bold')

        def autolabel(rects):
            # Attach a text label above each bar displaying its height
            for rect in rects:
                height = rect.get_height()
                print("getx", rect.get_x())
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%f' % float(height),
                        ha='center', va='bottom')

        autolabel(rects1)

        # Shows the figure.
        plt.show()

    def make_border(self, language, popularity):
        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]

        # plotting x_pos and popularity values to create bar chart
        plt.bar(x_pos, popularity, color='red', edgecolor='blue')

        # set the current tick locations and labels(language name)of the x-axis
        plt.xticks(x_pos, language)
        plt.title("Bar Chart with Border")
        # Shows the figure.
        plt.show()

    def increase_margin(self, language, popularity):
        # iterate through no.of language
        x_pos = [i for i, _ in enumerate(language)]
        # plotting x_pos and popularity values to create bar chart
        plt.bar(x_pos, popularity, color=(0.4, 0.6, 0.8, 1.0))

        # Rotation of the bars names
        plt.xticks(x_pos, language)

        # Custom the subplot layout
        plt.subplots_adjust(bottom=0.10, top=.4)
        plt.title("Bar Chart with increase in margin")
        # Shows the figure.
        plt.show()

    def specify_width_position(self, language, popularity):

        x_pos = [i for i, _ in enumerate(language)]

        plt.xticks(x_pos, language)

        # Select the width of each bar and their positions
        width = [0.1, 0.2, 0.5, 1.1, 0.2, 0.3]
        y_pos = [0, .8, 1.5, 3, 5, 6]

        # Create bars
        plt.bar(y_pos, popularity, width=width)
        plt.xticks(y_pos, language)
        plt.title("Bar Chart with width position")

        plt.show()
