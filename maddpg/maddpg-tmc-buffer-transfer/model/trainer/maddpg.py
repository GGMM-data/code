import numpy as np
import tensorflow as tf
import model.common.tf_util as U

from model.common.distributions import make_pdtype
from model import AgentTrainer
from model.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, p_scope, p_index, p_func, q_func, lstm_model, optimizer,
            grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, use_lstm=False):
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        # batch size的placeholder, []
        batch_size = tf.placeholder(tf.int32, shape=[], name="bs")
        # action placeholder
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]    # create distribtuions
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # observation placeholder
        obs_ph_n = make_obs_ph_n

        if use_lstm:
            observation_n = lstm_model(obs_ph_n, reuse=reuse, scope="lstm")
        else:
            observation_n = [tf.squeeze(o, 2) for o in obs_ph_n]
        p_input = observation_n[p_index]

    with tf.variable_scope(p_scope, reuse=tf.AUTO_REUSE):
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # act_pd.mode() #
        
        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()

    with tf.variable_scope(scope, reuse=reuse):
        q_input = tf.concat(observation_n + act_input_n, 1)     # 所有智能体的s和a, [batch_size, concat_dim]
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [batch_size], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]] + [batch_size], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]] + [batch_size], p)
        target_act = U.function(inputs=[obs_ph_n[p_index]] + [batch_size], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, lstm_model, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, use_lstm=False):
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        # batch size placeholder
        batch_size = tf.placeholder(tf.int32, shape=[], name="bs")
        # action placeholder
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]    # create distribtuions
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # observation placeholder
        obs_ph_n = make_obs_ph_n
        # target q placeholder
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        if use_lstm:
            observation_n = lstm_model(obs_ph_n, scope="lstm", reuse=reuse)
        else:
            observation_n = [tf.squeeze(o, 2) for o in obs_ph_n]
        if local_q_func:
            q_input = tf.concat([observation_n[q_index], act_ph_n[q_index]], 1)
        else:
            q_input = tf.concat(observation_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        if use_lstm:
            lstm_func_vars = U.scope_vars(U.absolute_scope_name("lstm"))  # lstm参数

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss   # + 1e-3 * q_reg

        if use_lstm:
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars + lstm_func_vars, grad_norm_clipping)
        else:
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [batch_size], outputs=loss,
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n + [batch_size], q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        target_q_values = U.function(obs_ph_n + act_ph_n + [batch_size], target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, env_name, name, model, lstm_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.env_name = env_name
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_shape = list(obs_shape_n[i])
            obs_shape.append(args.history_length)
            obs_ph_n.append(U.BatchInput((obs_shape), name="observation" + str(i)).get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.env_name + self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            lstm_model=lstm_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            use_lstm=False
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.env_name + self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_scope="common" + self.name,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            lstm_model=lstm_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            use_lstm=False
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args, obs_shape_n[0], act_space_n[0].n)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs, batch_size):
        return self.act(*(obs + batch_size))

    def experience(self, obs, act, rew, next_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, next_obs, float(done), terminal)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        obs_n = []  # 长度为n的list，list每隔元素为[batch_size, state_size, history_length]
        obs_next_n = []     # list的每个元素是[bath_size, action_dim]
        act_n = []
        for i in range(self.n):
            # todo
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample()
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample()

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](*([obs_next_n[i]] + [self.args.batch_size])) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n + [self.args.batch_size]))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q] + [self.args.batch_size]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n + [self.args.batch_size]))

        self.p_update()
        self.q_update()
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
