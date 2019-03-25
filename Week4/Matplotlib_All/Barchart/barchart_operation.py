import matplotlib.pyplot as plt
from Week4.Matplotlib_All.Utility.utility import UtilityClass
from Week4.Utility.Util import UtilClass


class MatplotlibBarChart:

    # class constructor
    def __init__(self):
        self.obj1 = UtilClass()

    # creates utility class object
    utility_obj = UtilityClass()

    # accept size of points you wanna accept
    size = utility_obj.accept_size()

    # accepts x axis values
    # print("Enter programming languages")
    language = utility_obj.accept_languages(size)
    print(language)

    # accepts y axis values
    print("Enter popularity")
    popularity = utility_obj.accept_popularity(size)
    print(popularity)

    # Set the x axis label
    plt.xlabel("Languages")

    # Set the y axis label
    plt.ylabel("Popularity")

    # Sets a title
    plt.title("Popularity of Programming Languages")

    def menu(self):
        print()
        print("1.Print output vertically")
        print("2.Print output horizontally")
        print("3.Bar with Uniform color")
        print("4.Bar with different color")
        print("5.Attach a text label above each bar displaying its popularity")
        print("6.Make blue border to each bar")
        print("9.Increase bottom margin")
        print("0.Exit")
        flag = False

        while not flag:
            try:
                choice = int(input("\nEnter your choice"))
                if choice >= 0 and choice <= 15:

                    if choice == 1:
                        self.obj1.draw_bar_vertically(self.language, self.popularity)

                    if choice == 2:
                        self.obj1.draw_bar_horizontally(self.language, self.popularity)

                    if choice == 3:
                        self.obj1.draw_bar_with_uniform_color(self.language, self.popularity)

                    if choice == 4:
                        self.obj1.draw_bar_with_diff_color(self.language, self.popularity)

                    if choice == 5:
                        self.obj1.attach_label(self.language, self.popularity)

                    if choice == 6:
                        self.obj1.make_border(self.language, self.popularity)

                    if choice == 8:
                        self.obj1.specify_width_position(self.language, self.popularity)

                    if choice == 9:
                        self.obj1.increase_margin(self.language, self.popularity)

                    if choice == 0:
                        flag = True
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input and Try again")


# creates class object
obj = MatplotlibBarChart()
flag = False

# calling method by using class object
obj.menu()
