# 显然下面的结果是错的
l = ['1', '2', 'c', 'a']
print(l)
# ['1', '2', 'c', 'a']
for i in l:
    if i in ['1', '2']:
        l.remove(i)
print(l)
# ['2', 'c', 'a']

l = ['1', '2', 'c', 'a']
print(l)
# ['1', '2', 'c', 'a']
l = [i for i in l if i not in ['1', '2']]
print(l)
# ['c', 'a']
