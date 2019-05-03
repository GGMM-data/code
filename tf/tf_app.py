import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('gpu', 'False', 'Use GPU?')

FLAGS = flags.FLAGS


def main(_):
    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print(FLAGS.model)


if __name__ == "__main__":
    tf.app.run(main)
