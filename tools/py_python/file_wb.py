import pickle

path = "./fig/test.pkl"
l = [1, 2, 3, 4]
with open(path, 'wb') as fp:
   pickle.dump(l, fp) 
