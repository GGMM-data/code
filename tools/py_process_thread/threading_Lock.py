import time
import threading
import random

hhhh = 100
lock = threading.Lock()

def add_number():
    global hhhh
    for i in range(100):
        with lock:
            hhhh += 1
            print("add: ", hhhh)
            time.sleep(0.015)
    
def subtract_number():
    global hhhh
    for i in range(100):
        with lock:
            hhhh -= 1
            print("subtract:", hhhh)
            time.sleep(0.015)
            # time.sleep(0.0001)
            # time.sleep(random.random())
 
 
job_list = []
job_list.append(threading.Thread(target=subtract_number, args=()))
job_list.append(threading.Thread(target=add_number, args=()))

for t in job_list:
    t.start()

for t in job_list:
    t.join()
 
print("Done")
