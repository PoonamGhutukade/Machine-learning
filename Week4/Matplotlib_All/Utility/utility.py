
class UtilityClass:

    def accept_size(self):
        n = int(input("Enter how many values you want to plot"))
        return n

    def CreateList(self, size):
        lst = []
        for i in range(size):
            words = int(input("Enter values"))
            lst.append(words)
        return lst

    def accept_languages(self, size):
        lst = []
        for i in range(size):
            words = input("Enter Programming language name")
            lst.append(words)
        return lst

    def accept_popularity(self, size):
        lst = []
        for i in range(size):
            words = float(input("Enter value"))
            lst.append(words)
        return lst

    def CheckInt(self, val):
        try:
            int(val)
            return True
        except Exception:
            return False
