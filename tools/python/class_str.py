class Tem(object):
  pass

class TemStr(object):
  def __str__(object):
    return 'foo'

class TemRepr(object):
  def __repr__(object):
    return 'foo'

class TemStrRepr(object):
  def __repr__(object):
    return 'foo'
  def __str__(object):
    return 'foo_str'

if __name__ == "__main__":
   tem = Tem()
   print(str(tem))
   print(repr(tem))
   tem_str = TemStr()
   print(str(tem_str))
   print(repr(tem_str))
   tem_repr = TemRepr()
   print(str(tem_repr))
   print(repr(tem_repr))
   tem_str_repr = TemStrRepr()
   print(str(tem_str_repr))
   print(repr(tem_str_repr))

