class Base(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


class A(Base):
    def __init__(self, name, age, gender):
        super(A, self).__init__(name, age)
        self.gender = gender

if __name__ == "__main__":
    a = A("mxx", 18, "man")
    print(a.age)
    print(a.name)
    print(a.gender)
