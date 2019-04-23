# a anonymous function mean that a function is without name.

# lambda arguments: expression
#   1.this function can have any number of arguments but only one expression, which is evaluated and returned
#   2.one if free to use lambda functions wherever functions objects are required
#   3.you need to keep in your knowledge that lambda functions are syntacitically restricted to a single expression
#   4.it has varous uses i particular fields of programming besides other types of expressions in functions

def cube(x):
   return x*x*x

g = lambda x: x*x*x
print(g(7))
print(cube(7))


# filter 
l = [1, 2, 3, 4, 5]
l1 = list(filter(lambda x: x*x, l))
print(l1)

# map 
l = [2, 4, 6, 8, 10]
m = list(map(lambda x: x*x, l))
print(m)


# 
add = lambda x,y: x+y
print(add(1,3))
