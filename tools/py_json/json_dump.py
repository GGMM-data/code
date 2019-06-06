import json

person = '{"name": "Bob", "languages": ["Python", "Java"]}'
print(type(person))


with open('person_dump.json', "w+") as f:
  data = json.dump(person, f)
