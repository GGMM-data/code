import tensorflow as tf
import gym
import multiprocessing as mp
import numpy as np
import threading

workers_num = mp.cpu_count()
env_name = "CartPole-v0"
global_net_scope = 'GLOBAL_NET'
lr_actor = 0.001
lr_critic = 0.001
beta = 0.001
update_global_number = 10
gamma = 0.9

global_episode_number = 0
max_global_episode_number = 1000

global_episode_reward = []

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class ACNet(object):
    def __init__(self, scope, global_ac=None):
        if scope == global_net_scope:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
                self.actor_params, self.critic_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
                self.action = tf.placeholder(tf.int32, [None, ], 'action')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'target_value')

                self.action_prob, self.state_value, self.actor_params, self.critic_params = self._build_net(scope)
                td_error = tf.subtract(self.target_value, self.state_value, name='td_error')
            with tf.name_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(td_error))

            with tf.name_scope('actor_loss'):
                log_prob = tf.reduce_sum(tf.log(self.action_prob + 1e-5) * tf.one_hot(self.action, action_dim, dtype=tf.float32), axis=1, keep_dims=True)
                exp_v = log_prob * tf.stop_gradient(td_error)
                entropy = - tf.reduce_sum(self.action_prob + 1e-5, axis=1, keep_dims=True)
                self.exp_v = beta * entropy + exp_v
                self.actor_loss = tf.reduce_mean(- self.exp_v)

            with tf.name_scope('local_grad'):
                self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)
                
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_actor_params_op = [local_params.assign(global_params) for local_params, global_params in zip(self.actor_params, global_ac.actor_params)]
                    self.pull_critic_params_op = [local_params.assign(global_params) for local_params, global_params in zip(self.critic_params, global_ac.critic_params)]

                with tf.name_scope('push'):
                    self.update_actor_params_op = optimizer_actor.apply_gradients(zip(self.actor_grads, global_ac.actor_params))
                    self.update_critic_params_op = optimizer_critic.apply_gradients(zip(self.critic_grads, global_ac.critic_params))

    def _build_net(self, scope):
        w_initializer = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope('actor'):
            actor_layer = tf.layers.dense(self.state, 200, tf.nn.relu, kernel_initializer=w_initializer, name='actor_layer' )
            action_prob = tf.layers.dense(actor_layer, action_dim, tf.nn.softmax, kernel_initializer=w_initializer, name='action_probability')

        with tf.variable_scope('critic'):
            critic_layer = tf.layers.dense(self.state, 100, tf.nn.relu, kernel_initializer=w_initializer, name='critic_layer')
            state_value = tf.layers.dense(critic_layer, 1, kernel_initializer=w_initializer, name='state_value')

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')
        return action_prob, state_value, actor_params, critic_params

    def update_global(self, feed_dict):
        sess.run([self.update_actor_params_op, self.update_critic_params_op], feed_dict)
    
    def pull_global(self):
        sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, state):
        prob_weights = sess.run(self.action_prob, feed_dict={self.state: state[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) #

        return action

class Worker(object):
    def __init__(self, name, global_ac):
        self.env = gym.make(env_name).unwrapped
        self.name = name
        self.ac = ACNet(name, global_ac)

    def work(self):
        global global_episode_number, global_episode_reward
        state_list = []
        action_list = []
        reward_list = []
        total_step = 0
        while not coord.should_stop() and global_episode_number < max_global_episode_number:
            state = self.env.reset()
            returns = 0
            while True:
                action = self.ac.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    r = -5
                returns += reward

                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                if total_step % update_global_number == 0 or done:
                    if done:
                        next_state_value = 0
                    else:
                        next_state_value = sess.run(self.ac.state_value, {self.ac.state: next_state[None, :]})[0, 0]

                    target_value = []
                    for r in reward_list[::-1]:
                        next_state_value = r + gamma * next_state_value
                        target_value.append(next_state_value)
                    target_value.reverse()
                    state_array, action_array, target_value_array = np.vstack(state_list), np.array(action_list), np.vstack(target_value)
                    feed_dict = {
                        self.ac.state: state_array,
                        self.ac.action: action_array,
                        self.ac.target_value: target_value_array,
                    }
                    self.ac.update_global(feed_dict)
                    state_list, action_list, reward_list = [], [], []
                    # 这里pull一下是解决cpu切换到其他线程将parameter server的参数更新之后该线程的参数和parameter server的参数不一样的问题。
                    self.ac.pull_global()

                state = next_state
                total_step += 1
                if done:
                    global_episode_reward.append(returns)
                    print(self.name, "episode: ", global_episode_number, "episode reward: ", global_episode_reward[-1])
                    with lock:
                        global_episode_number += 1
                    break

if __name__ == "__main__":
    lock = threading.Lock()
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            optimizer_actor = tf.train.RMSPropOptimizer(lr_actor, name='RMSProp_actor')
            optimizer_critic = tf.train.RMSPropOptimizer(lr_critic, name='RMSProp_critic')
            global_ac = ACNet(global_net_scope)
            workers = []
            for i in range(workers_num):
                name = "work_%i" % i
                workers.append(Worker(name, global_ac))

        coord = tf.train.Coordinator()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)

