import itertools



# 1.count 
# count will iterate infintely times, the parameter is when to begin
for t in itertools.count(3):
  print(t)
  if t > 10:
    break

# 2.cycle
c = itertools.cycle('ABC')
count = 0
for i in c:
  print(i) 
  if count > 10:
    break
  count += 1

# 3.repeat
# repeat 20 times
r = itertools.repeat(1, 5)
for i in r:
  print(i)

# repeat infintely times
r = itertools.repeat(1)
count = 0
for i in r:
  print(i)
  if count > 7:
    break
  count += 1

