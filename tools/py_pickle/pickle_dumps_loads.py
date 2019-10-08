import pickle

dictionary = {"name": "mxx", "age": 23}

s = pickle.dumps(dictionary)
print(s)
print(type(s))

b = pickle.loads(s)
print(b)
print(type(b))
