class MyClass:
  """This is my class __doc__"""
  class_name = "cllll"
  def __init__(self, test=None):
     self.test = test
  pass

if __name__ == "__main__":

  cl = MyClass()
  print(cl.__dict__)
  print(cl.__doc__)
  print(cl.__module__)
