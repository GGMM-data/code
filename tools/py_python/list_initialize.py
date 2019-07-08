length = 10
v1 = [[]]*length
for i in range(length):
    v1[i].append(i)
print("v1")
print(v1)

v2 = [[] for _ in range(length)]
for i in range(length):
    v2[i].append(i)

print("v2")
print(v2)

