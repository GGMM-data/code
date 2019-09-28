import tensorflow as tf

#saver = tf.train.Saver()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("saver1.ckpt.meta")
    print(sess.graph)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var)
    print(sess.graph.get_all_collection_keys())
    print(sess.graph.collections)
    print(sess.graph.get_all_collection_keys())
