import threading
import socket
import os

def do_connect(website):
    s = socket.socket()
    info = s.connect((website, 80))  # drop the GIL
    print(type(info))
    print(info)
    print(os.getpid())

websites = ['python.org', 'baidu.com']

for i in range(len(websites)):
    t = threading.Thread(target=do_connect, args=(websites[i],))
    t.start()

