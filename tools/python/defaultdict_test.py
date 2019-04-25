from collections import defaultdict

dd = defaultdict(dict)
print(dd)

m = dd['a']
m['step'] = 1
m['exp'] = 3
print(dd)

m = dd['b']
m['step'] = 1
m['exp'] = 3
print(dd)
