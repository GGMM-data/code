import tensorflow as tf

g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")
  cc = tf.Variable([2])

  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)
print(sess_1.graph)
print(sess_2.graph)
print(tf.get_default_graph())
#for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(var)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
