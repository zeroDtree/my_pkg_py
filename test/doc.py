class A:
    def __init__(self, g):
        self.g = g

    def f(self, a):
        print(a)


def gg(a):
    print(a)


m = A(gg)

m.g(1)

print(m.f)
print(m.g)
