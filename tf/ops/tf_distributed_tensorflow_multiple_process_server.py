import tensorflow as tf

c = tf.constant("hello, distributed tensorflow!")
server = tf.train.Server.create_local_server()
print(type(server))
# <class 'tensorflow.python.training.server_lib.Server'>
print(server)
# <tensorflow.python.training.server_lib.Server object at 0x7fdea6306f60>
print(type(server.target))
# <class 'str'>
print(server.target)
# grpc://localhost:42001

sess = tf.Session(server.target)
result = sess.run(c)
print(result)
