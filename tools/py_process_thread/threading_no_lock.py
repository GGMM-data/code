import time
import threading
import random

hhhh = 0
lock = threading.Lock()

def add_number():
    global hhhh
    for i in range(10):
        print("before add: ", i, hhhh)
        hhhh += 1
        print("after add: ", i, hhhh)
    
def subtract_number():
    global hhhh
    for i in range(10):
        print("before substract: ", i, hhhh)
        hhhh -= 1
        print("after subtract: ", i, hhhh)
 
 
job_list = []
job_list.append(threading.Thread(target=subtract_number, args=()))
job_list.append(threading.Thread(target=add_number, args=()))

for t in job_list:
    t.start()

for t in job_list:
    t.join()
 
print("Done")
