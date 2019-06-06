import json

person = '{"name": "Bob", "languages": ["Python", "Java"]}'
print(type(person))

data = json.dumps(person)
print(data)
print(data['name'])
