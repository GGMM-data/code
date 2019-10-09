from importlib import import_module

defa = import_module('.'.join(["default"]))
# gettattr(defa, 'info') 是 a function
attr = getattr(defa, 'info')
print(attr())
