
import re
from Week4.Utility.Util import UtilClass


class Matplotlib:
    # class constructor
    def __init__(self):
        self.obj1 = UtilClass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Draw a line with suitable label on the x axis, y axis and a title""\n"
                      "2. Draw a line values taken from a text file and a title.  ""\n"
                      "3. Draw line charts of the financial data by using csv file\n"
                      "4. Plot two or more lines on same plot with suitable legends of each line""\n"
                      "5. Plot two or more lines with legends, different widths and colors""\n"
                      "6. Plot two or more lines with different styles""\n"
                      "7. Plot two or more lines and set the line markers.""\n"
                      "8. Display the current axis limits values and set new axis values""\n"
                      "9. Plot quantities which have an x and y position.""\n"
                      "10. Plot several lines with different format styles in one command using arrays""\n"
                      "11. Create multiple types of charts ""\n"
                      "12. Customized the grid lines with linestyle -, width .5. and color blue""\n"
                      "13. Customized the grid lines with rendering with a larger grid and a smaller grid""\n"
                      "14. Create multiple plots""\n"
                      
                      
                      "5. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        """1.Program to draw a line with suitable label in the x axis, y axis and title,
                         2. Draw a line using given axis values with suitable label in the x axis , y axis and a title"""

                        size = input("Enter the size for X and Y axis:")
                        x_axis = self.obj1.creatlist(size)
                        print(x_axis)
                        y_axis = self.obj1.creatlist(size)
                        print(y_axis)

                        self.obj1.draw_line_matplotlib(x_axis, y_axis)

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        """3. Draw a line using given axis values taken from a text file, 
                        with suitable label in the x axis, y axis and a title."""
                        self.obj1.read_text_file()

                        print("_______________________________________________________________________________")

                    elif choice == 3:
                        """ Write a Python program to draw line charts of the financial data of Alphabet Inc. 
                        between October 3, 2016 to October 7, 2016. Sample Financial data (fdata.csv)"""
                        self.obj1.read_data()
                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        """Plot two or more lines on same plot with suitable legends of each line."""
                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.draw__multiple_line(x1_axis, y1_axis,x2_axis, y2_axis)
                        # self.obj1.draw_line_matplotlib(x1_axis, y1_axis)
                        # self.obj1.draw_line_matplotlib(x1_axis, y1_axis)

                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        """plot two or more lines with legends, different widths and colors"""
                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.draw__line_with_color(x1_axis, y1_axis, x2_axis, y2_axis)
                        print("_______________________________________________________________________________")

                    elif choice == 6:
                        """plot two or more lines with different styles"""
                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.plot_line_with_style(x1_axis, y1_axis, x2_axis, y2_axis)
                        print("_______________________________________________________________________________")

                    elif choice == 7:
                        """plot two or more lines and set the line markers."""
                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.line_marker(x1_axis, y1_axis, x2_axis, y2_axis)
                        print("_______________________________________________________________________________")

                    elif choice == 8:
                        """display the current axis limits values and set new axis values"""
                        self.obj1.type_chart()
                        print("_______________________________________________________________________________")

                    elif choice == 9:
                        """plot quantities which have an x and y position."""

                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.plot_quantities(x1_axis, y1_axis, x2_axis, y2_axis)

                        print("_______________________________________________________________________________")
                    elif choice == 10:
                        """ plot several lines with different format styles in one command using arrays"""
                        self.obj1.draw_line_with_diff_formats_witharray()
                        print("_______________________________________________________________________________")
                    elif choice == 11:
                        """create multiple types of charts"""
                        print("_______________________________________________________________________________")

                    elif choice == 12:
                        """Write a Python program to display the grid and draw line charts of the closing value of 
                        Alphabet Inc. between October 3, 2016 to October 7, 2016. Customized the grid lines with
                         linestyle -, width .5. and color blue."""
                        self.obj1.custom_grid()
                        print("_______________________________________________________________________________")

                    elif choice == 13:
                        """Write a Python program to display the grid and draw line charts of the closing value of 
                        Alphabet Inc. between October 3, 2016 to October 7, 2016. Customized the grid lines with 
                        rendering with a larger grid (major grid) and a smaller grid (minor grid).Turn on the grid 
                        but turn off ticks. """
                        self.obj1.large_small_grid()

                        print("_______________________________________________________________________________")

                    elif choice == 14:
                        """program to create multiple plots"""

                        size = input("Enter the size for 1st line:")
                        x1_axis = self.obj1.creatlist(size)
                        print(x1_axis)
                        y1_axis = self.obj1.creatlist(size)
                        print(y1_axis)
                        size = input("Enter the size for 2nd line:")
                        x2_axis = self.obj1.creatlist(size)
                        print(x2_axis)
                        y2_axis = self.obj1.creatlist(size)
                        print(y2_axis)

                        self.obj1.create_subplot(x1_axis, y1_axis, x2_axis, y2_axis)
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
obj = Matplotlib()
obj.calling()

