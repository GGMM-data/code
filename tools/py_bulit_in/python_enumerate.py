a = ['a', 'b', 'c']
b = ['s', 'm', 'n']

# 1, enumerate list
for element in enumerate(a):
   print(type(element))
   print(element[0])
   print(element[1])

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons)))
print(list(enumerate(seasons, start=1)))
for element in enumerate(seasons):
   print(element)


# 2. enumerate two lists
for idx, (e1, e2) in enumerate(zip(a,b)):
    print(idx, e1, e2)

# https://stackoverflow.com/questions/16326853/enumerate-two-python-lists-simultaneously
