
def p_func(name, content):
    print(name)
    return a_func


def a_func(content):
    print(content)
    pass


class A:
    def __init__(self, name):
        self.name = name
        self.p_train = None
        self.count = 0


class B:
    def __init__(self):
        self.name = "B"

    def add_C(self, A):
        self.func = p_func(A.name)
        A.p_train = self.func


if __name__ == "__main__":
    a = [A("a1"), A("a2"), A("a3"), A("a3")]
    b = B()
    for k  in range(2):
        for i in range(4):
            b.add_C(a[i])
            print(a[i].p_train)

