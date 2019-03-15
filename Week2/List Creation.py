
def listadd():
    try:
        arrr = list()
        size = input("Enter the size of an tuple for same Dt:")
        # we have to typecast num to compare with length of string
        num2 = int(size)
        res = 0
        resmul = 0
        # checking enter value is only digit or not
        if size.isdigit():
            print("Enter the elements: ")
            for ele in range(num2):
                res += int(input())
                arrr.append(res)
            return res

        else:
            raise ValueError
    except ValueError:
        print("Enter valid number: ")


res = listadd()
print("Addition of List items: ", res)

