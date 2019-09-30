import tensorflow as tf


def loadmodel(f=None):
    # 1. define a session
    with tf.Session() as sess:
        # 2.1 checkpoint path
        ckpt_path = "model.ckpt"
        # 2.2 find the lastest checkpoint path
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print(ckpt)
        # 3.load
        if ckpt and ckpt.model_checkpoint_path:
            # 3.1 coumpute graph
            saver = tf.train.import_meta_graph(ckpt+".meta", clear_devices=True)
            # 3.2 load weights
            saver.restore(sess, ckpt.model_checkpoint_path) 
        print(sess.graph.collections)
        x_op = sess.graph.get_tensor_by_name("Placeholder:0")
    
        print(x_op)


if __name__ == "__main__":

    loadmodel()
