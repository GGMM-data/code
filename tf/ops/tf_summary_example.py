import argparse
import os
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
  # 读取数据
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  # 生成参数函数
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  #  定义summary操作
  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 记录网络参数的一些信息
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        # 调用自定义函数，记录weights相关信息。
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        # 调用自定义函数，记录biases相关信息。
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        # 记录？？
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      # 记录激活函数？？
      tf.summary.histogram('activations', activations)
      return activations

  # 构建网络
  # 第一个隐藏层
  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    # 记录dropout概率
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # 第二层
  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
  
  # 计算损失函数。
  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y)
  # 记录loss
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # 记录accuracy
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  # 创建两个FileWriter写入event 文件。
  # 一个用于训练，一个用于测试
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  # 初始化变量
  tf.global_variables_initializer().run()

"""
  ## 查看graph的一些collection
  为什么在这个位置加，不在开头或者结尾加这一部分，在开头，还没有构建图，在结尾，还需要运行整个程序之后才能看到。
  print("=====trainable=====")
  for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
     print(var)
  print("=====global=====")
  for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
     print(var)
  print("=====model=====")
  for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
     print(var)
  print("=====summaries=====")
  for var in tf.get_collection(tf.GraphKeys.SUMMARIES):
     print(var)
  return 

# 输出
=====trainable=====
<tf.Variable 'layer1/weights/Variable:0' shape=(784, 500) dtype=float32_ref>
<tf.Variable 'layer1/biases/Variable:0' shape=(500,) dtype=float32_ref>
<tf.Variable 'layer2/weights/Variable:0' shape=(500, 10) dtype=float32_ref>
<tf.Variable 'layer2/biases/Variable:0' shape=(10,) dtype=float32_ref>
=====global=====
<tf.Variable 'layer1/weights/Variable:0' shape=(784, 500) dtype=float32_ref>
<tf.Variable 'layer1/biases/Variable:0' shape=(500,) dtype=float32_ref>
<tf.Variable 'layer2/weights/Variable:0' shape=(500, 10) dtype=float32_ref>
<tf.Variable 'layer2/biases/Variable:0' shape=(10,) dtype=float32_ref>
## 下面10行都是Adam Optimizer生成的Variable，如果把Adam Optimizer注释掉，那么就不会有下面这些行了，只会有上面四行，和TRAINABLE_VARIABLES内容一样。
<tf.Variable 'train/beta1_power:0' shape=() dtype=float32_ref>
<tf.Variable 'train/beta2_power:0' shape=() dtype=float32_ref>
<tf.Variable 'layer1/weights/Variable/Adam:0' shape=(784, 500) dtype=float32_ref>
<tf.Variable 'layer1/weights/Variable/Adam_1:0' shape=(784, 500) dtype=float32_ref>
<tf.Variable 'layer1/biases/Variable/Adam:0' shape=(500,) dtype=float32_ref>
<tf.Variable 'layer1/biases/Variable/Adam_1:0' shape=(500,) dtype=float32_ref>
<tf.Variable 'layer2/weights/Variable/Adam:0' shape=(500, 10) dtype=float32_ref>
<tf.Variable 'layer2/weights/Variable/Adam_1:0' shape=(500, 10) dtype=float32_ref>
<tf.Variable 'layer2/biases/Variable/Adam:0' shape=(10,) dtype=float32_ref>
<tf.Variable 'layer2/biases/Variable/Adam_1:0' shape=(10,) dtype=float32_ref>
=====model=====
=====summaries=====
Tensor("input_reshape/input:0", shape=(), dtype=string)
Tensor("layer1/weights/summaries/mean_1:0", shape=(), dtype=string)
Tensor("layer1/weights/summaries/stddev_1:0", shape=(), dtype=string)
Tensor("layer1/weights/summaries/max_1:0", shape=(), dtype=string)
Tensor("layer1/weights/summaries/min_1:0", shape=(), dtype=string)
Tensor("layer1/weights/summaries/histogram:0", shape=(), dtype=string)
Tensor("layer1/biases/summaries/mean_1:0", shape=(), dtype=string)
Tensor("layer1/biases/summaries/stddev_1:0", shape=(), dtype=string)
Tensor("layer1/biases/summaries/max_1:0", shape=(), dtype=string)
Tensor("layer1/biases/summaries/min_1:0", shape=(), dtype=string)
Tensor("layer1/biases/summaries/histogram:0", shape=(), dtype=string)
Tensor("layer1/Wx_plus_b/pre_activations:0", shape=(), dtype=string)
Tensor("layer1/activations:0", shape=(), dtype=string)
Tensor("dropout/dropout_keep_probability:0", shape=(), dtype=string)
Tensor("layer2/weights/summaries/mean_1:0", shape=(), dtype=string)
Tensor("layer2/weights/summaries/stddev_1:0", shape=(), dtype=string)
Tensor("layer2/weights/summaries/max_1:0", shape=(), dtype=string)
Tensor("layer2/weights/summaries/min_1:0", shape=(), dtype=string)
Tensor("layer2/weights/summaries/histogram:0", shape=(), dtype=string)
Tensor("layer2/biases/summaries/mean_1:0", shape=(), dtype=string)
Tensor("layer2/biases/summaries/stddev_1:0", shape=(), dtype=string)
Tensor("layer2/biases/summaries/max_1:0", shape=(), dtype=string)
Tensor("layer2/biases/summaries/min_1:0", shape=(), dtype=string)
Tensor("layer2/biases/summaries/histogram:0", shape=(), dtype=string)
Tensor("layer2/Wx_plus_b/pre_activations:0", shape=(), dtype=string)
Tensor("layer2/activations:0", shape=(), dtype=string)
Tensor("cross_entropy_1:0", shape=(), dtype=string)
Tensor("accuracy_1:0", shape=(), dtype=string)
  """

  # 生成训练数据
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  # 训练
  for i in range(FLAGS.max_steps):
    # 每隔10步保存一下测试数据
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      # 记录测试信息
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      # 每隔100步除了记录summary还要记录run_metadata
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        # 每一步都记录一下summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  # 关闭FileWriter
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  # 传递参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
      default=os.path.join('tmp',
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      # default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
      default=os.path.join('tmp',
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
