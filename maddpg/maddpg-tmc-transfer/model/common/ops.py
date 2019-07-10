import tensorflow as tf
import os
import sys

sys.path.append(os.getcwd()+ "/../")

import model.common.tf_util as U
from model.common.distributions import make_pdtype


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


# 优化critic用的是MSE,怎么用到actor,actor用来选择action
def q_train(make_obs_ph_n, act_space_n, q_index, q_func, lstm_model, optimizer, args, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64, use_lstm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # ===================q network开始建图=================
        # set up placeholders
        # batch size placeholder
        batch_size = tf.placeholder(tf.int32, shape=[], name="bs")
        # action placeholder
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]    # 创建分布
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # observation placeholder
        obs_ph_n = make_obs_ph_n  # 创建obs placeholder
        # target q placeholder
        target_ph = tf.placeholder(tf.float32, [None], name="target")  # 在运行时计算，然后传入，只跟loss有关
        
        # 在这里进行dimension reduction
        if use_lstm:
            observation_n = lstm_model(obs_ph_n, scope="lstm", reuse=reuse)
        else:
            observation_n = [tf.squeeze(o, 2) for o in obs_ph_n]
        # 所有智能体的obs和action
        if local_q_func:
            q_input = tf.concat([observation_n[q_index], act_ph_n[q_index]], 1)
        else:
            q_input = tf.concat(observation_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]  # 计算q值
        
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))  # q network网络参数
        if use_lstm:
            lstm_func_vars = U.scope_vars(U.absolute_scope_name("lstm"))  # lstm参数
        
        q_loss = tf.reduce_mean(tf.square(q - target_ph))  # mse loss
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg
        
        if use_lstm:
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars + lstm_func_vars, grad_norm_clipping)
        else:
            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        # ===============q network建图结束=====================
        
        # 创建可调用函数
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + [batch_size], outputs=loss,
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n + [batch_size], q)
        
        # ==================target q network建图===============
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        # ===================target q network建图结束======================
        
        # 创建可调用函数
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        target_q_values = U.function(obs_ph_n + act_ph_n + [batch_size], target_q)
        
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


# 创建p_func和lstm_func,target_p_func
def p_act(make_obs_ph_n, act_space_n, p_index, p_func, lstm_model,
            num_units=64, scope="trainer", reuse=None, use_lstm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # ============p network建图=================
        # batch size的placeholder, []
        batch_size = tf.placeholder(tf.int32, shape=[], name="bs")
        # observation placeholder, list of [batch_size, dim, time_step]
        obs_ph_n = make_obs_ph_n

        # action distribution
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 创建action的分布用来采样

        if use_lstm:
            observation_n = lstm_model(obs_ph_n, reuse=reuse, scope="lstm")
        else:
            observation_n = [tf.squeeze(o, 2) for o in obs_ph_n]
        # 当前智能体的局部obs, [batch_size, state_dim]
        p_input = observation_n[p_index]
        
        # 计算局部p值，最后用来产生action, [batch_size, action_dim]
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units, reuse=reuse)
        # p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        act_pd = act_pdtype_n[p_index].pdfromflat(p)  #
        act_sample = act_pd.sample()  # [batch_size, action_dim]
        # ============p network建图结束=================
        
        # 采样aciton的函数调用
        act = U.function(inputs=[obs_ph_n[p_index]] + [batch_size], outputs=act_sample)
        # p value的函数调用
        p_values = U.function([obs_ph_n[p_index]] + [batch_size], p)
        
        # ============target p network建图=================
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", reuse=reuse,
                          num_units=num_units)
        # target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        target_act_pd = act_pdtype_n[p_index].pdfromflat(target_p)  # target action, [batch_size, action_dim]
        target_act_sample = target_act_pd.sample()
        # ============p target network建图结束=================

        # target action的函数调用
        target_act = U.function(inputs=[obs_ph_n[p_index]] + [batch_size], outputs=target_act_sample)
        return act,  {'p_values': p_values, 'target_act': target_act}
        # update_target_p = make_update_exp(p_func_vars, target_p_func_vars)    # 更新目标p网络的参数
        # return act, update_target_p,  {'p_values': p_values, 'target_act': target_act}


# 优化actor用的是policy gradient,怎么用到critic,把所有任务的performance measure加起来？？？把scope传进来就ok了。
def p_train(make_obs_ph_n, act_space_n, p_scope, p_index, p_func, q_func, lstm_model, optimizer,
            grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, use_lstm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholder
        # batch size placeholder
        batch_size = tf.placeholder(tf.int32, shape=[], name="bs")  # batch size的placeholder, []
        # action placeholder, list of [batch_size, action_dim]
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 创建action的分布用来采样
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # observation placeholder
        obs_ph_n = make_obs_ph_n    # 创建observation的placeholder, list of [batch_size, state_dim, time_step]
        
        if use_lstm:
            observation_n = lstm_model(obs_ph_n, reuse=reuse, scope="lstm")
            lstm_vars = U.scope_vars(U.absolute_scope_name("lstm"))
        else:
            observation_n = [tf.squeeze(o, 2) for o in obs_ph_n]    # 所有智能体的obs, list of [batch_size, state_dim]
        p_input = observation_n[p_index]    # 当前智能体的局部obs, [batch_size, state_dim]
        
    # p是多个actor公用的，q是每一个critic有一个
    with tf.variable_scope(p_scope, reuse=reuse):
        # 计算局部p值，最后用来产生action, [batch_size, action_dim]
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", reuse=True, num_units=num_units)
        # p函数的参数
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution,Pd.logits
        act_pd = act_pdtype_n[p_index].pdfromflat(p)  #
        # act_sample = act_pd.sample()    # [batch_size, action_dim]
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))  # [None]
        # 更新action
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # [batch_size, action]

        # 目标p值的参数
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)  # 函数调用
        
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        q_input = tf.concat(observation_n + act_input_n, 1)     # 所有智能体的s和a, [batch_size, concat_dim]
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]  # 计算Q(s,a), [batch_size,]
        
        # loss
        pg_loss = - tf.reduce_mean(q)    # policy gradient loss ???
        loss = pg_loss + p_reg * 1e-3   # 使用每一个critic计算的loss都是不同的，第一次需要建图，以后就不需要了
        
        # p网络的优化器。
        if use_lstm:
            optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars + lstm_vars, grad_norm_clipping)
        else:
            optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        # ============p network建图结束=================

        # 创建可以调用的函数，就是往里面喂数据
        # train的调用函数，输入必须是list，
        train = U.function(inputs=obs_ph_n + act_ph_n + [batch_size], outputs=loss, updates=[optimize_expr])
        return train, update_target_p
