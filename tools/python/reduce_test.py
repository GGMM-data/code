from functools import reduce

# reduce(function, iterable, [initializer])
# 第一个参数是函数，第二个参数是一个可迭代的变量，第三个参数是初始值（可选）。
# reduce 函数对第二个参数使用第一个函数参数累计计算。
def add(x, y):
  return x + y

print(reduce(add, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

def minus(x, y):
  return x - y 

print(reduce(minus, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
