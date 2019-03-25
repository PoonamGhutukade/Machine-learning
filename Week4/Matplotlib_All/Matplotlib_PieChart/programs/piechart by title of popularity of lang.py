from Week4.Matplotlib_All.Matplotlib_PieChart.Utility.piechart_utility import validate_num, create_list_all
import matplotlib.pyplot as plt


# class to perform graphical representation of data using matplotlib pie chart
class ChartByTitlePopularity:
    choice = 0

    def pie_chart(self):
        print()
        print("1. Create a pie chart with a title of the popularity of programming Languages.")
        print("2. Exit")
        print()
        while True:
            try:
                print()
                # accept choice from user
                self.choice = input("Enter choice : ")
                # validate choice number
                valid_choice = validate_num(self.choice)
                if valid_choice:
                    choice = int(self.choice)
                    if choice == 1:
                        print("Enter programming languages (5):")
                        # create list of 5 languages
                        lang = create_list_all(5)
                        print("Enter popularity of that language:")
                        # list of popularity
                        popularity = create_list_all(5)
                        # explode 1st slice
                        explode = (0.1, 0, 0, 0, 0)
                        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

                        # plot
                        plt.pie(popularity, explode=explode, labels=lang, colors=colors, autopct='%1.1f%%', shadow=True,
                                startangle=140)
                        plt.axis('equal')

                        # set the title
                        plt.title("Popularity of Programming Language\n" + "Worldwide, Mar 2019 compared to a year ago",
                                  bbox={'facecolor': '0.8', 'pad': 5})
                        plt.show()
                    elif choice == 2:
                        exit()
                    else:
                        print("Enter valid choice")
                else:
                    print("Enter only numbers")
            except Exception as e:
                print(e)


# obj of class created
object = ChartByTitlePopularity()
object.pie_chart()
