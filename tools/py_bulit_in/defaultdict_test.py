from collections import defaultdict

print("=================defaultdict(dict)================")
ddd = defaultdict(dict)
print(ddd)

m = ddd['a']
m['step'] = 1
m['exp'] = 3
print(type(m))
print(ddd)

m = ddd['b']
m['step'] = 1
m['exp'] = 3
print(ddd)

print("=================defaultdict(list)================")
ddl = defaultdict(list)
print(ddl)
m = ddl['a']
print(type(m))
m.append(3)
m.append('hhhh')
print(ddl)
