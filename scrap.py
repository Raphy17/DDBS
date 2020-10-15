a = [1, 2, 4, 6, 34, 7]
b = [1, 3, 2, 4, 87]


class Test():

    def __init__(self):
        self.a, self.b = self.test1()

    def test1(self):
        return (3, 5)



x = Test()
print(x.b)