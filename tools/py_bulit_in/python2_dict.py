m_dict = {'a': 10, 'b': 20}
print(m_dict)
m_dict.update({})
m_dict.update({"name": "mxx"})
print(m_dict)

values = m_dict.values()
print(type(values))
print(values)
print("\n")

items = m_dict.items()
print(type(items))
print(items)
print("\n")

keys = m_dict.keys()
print(type(keys))
print(keys)
print("\n")

l_values = list(values)
print(type(l_values))
print(l_values)
