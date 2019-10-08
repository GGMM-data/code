import pickle

dictionary = {"name": "mxx", "age": 23}

with open("test.txt", 'wb') as f:
    pickle.dump(dictionary, f)

with open("test.txt", 'rb') as f:
    b = pickle.load(f)

print(b)
print(type(b))
