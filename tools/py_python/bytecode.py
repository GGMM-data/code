import dis

n = 0

def foo():
    global n
    n += 1


print(dis.dis(foo))
