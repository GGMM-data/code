a = ['a', 'b', 'c']
for i in enumerate(a):
   print(type(i))
   print(i[0])
   print(i[1])
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons)))
print(list(enumerate(seasons, start=1)))
for i in enumerate(seasons):
   print(i)
