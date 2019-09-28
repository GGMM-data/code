import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    variable = tf.Variable(10, name="foo")
    initialize = tf.global_variables_initializer()
    assign = variable.assign(12)

with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    sess.run(assign)
    print(sess.run(variable))

with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    print(sess.run(variable))

print(tf.get_default_session())
