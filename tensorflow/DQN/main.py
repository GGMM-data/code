import tensorflow as tf
from myown.tf.DQN.config import get_config
from myown.tf.DQN.dqn.replaybuffer import ReplayBuffer
from myown.tf.DQN.dqn.environment import Environment
from myown.tf.DQN.dqn.agent import Agent
import time

def parse_args():
    flags = tf.app.flags
    flags.DEFINE_string('model', 'm1', 'Type of model')
    flags.DEFINE_integer('scale', '10', 'Type of model')
    FLAGS = flags.FLAGS
    return FLAGS


def main(_):
    print(config.scale)
    print(config.memory_size)
    replay_buffer = ReplayBuffer(config)
    sess = tf.Session()

    x = tf.Variable(tf.truncated_normal(shape=[5]))
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
    agent = Agent(config, sess=sess)
    agent.save_model(step=1)
    agent.load_model()

    print(agent.checkpoint_dir)
    env = Environment(config)

    # env.new_random_game()
    # while True:
    #     time.sleep(2)

        # print("hh")
        # env._random_step()
        # env.after_act()


if __name__ == "__main__":
    FLAGS = parse_args()
    config = get_config(FLAGS)
    main(config)
