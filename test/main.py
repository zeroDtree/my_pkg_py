import sys

class MyClass1:
    pass

class MyClass2:
    __slots__ = ('a', 'b')

obj1 = MyClass1()
obj2 = MyClass2()

print(sys.getsizeof(obj1))  # 输出对象占用的内存大小
print(sys.getsizeof(obj2))  # 输出对象占用的内存大小

print(obj1.__dict__)
print(obj2.__dict__)

from torch import Tensor