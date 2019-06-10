import json

person = '{"name": "Bob", "languages": ["Python", "Java"]}'
print(type(person))

person_dict = json.loads(person)
print(type(person_dict))
