import gym
import numpy as np
from gym import spaces
import networkx as nx
from multiagent.uav.flag import FLAGS


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        # custom parameters for uav begin-------------------------------------------------------------------------------
        self.uav = getattr(FLAGS, 'num_uav')
        self.size = getattr(FLAGS, 'size_map')
        self.radius = getattr(FLAGS, 'radius') ** 2
        self.max_epoch = getattr(FLAGS, 'max_epoch')
        self.map_scale_rate = getattr(FLAGS, 'map_scale_rate')

        self.PoI = []
        base = - (self.size-1)/2
        for i in range(self.size):
            for j in range(self.size):
                self.PoI.append([base + i, base + j])
        self.poi_array = np.array(self.PoI)     # [size * size, 2]
        self.M = np.zeros((self.size, self.size))
        self.final = np.zeros((self.size, self.size), dtype=np.int64)
        self.state = []

        for a in world.agents:
            location_tem = [a.state.p_pos[0] * self.map_scale_rate, a.state.p_pos[1] * self.map_scale_rate]
            self.state.append(location_tem)
        # energy
        self.energy = np.zeros(self.uav)
        self.jain_index = 0
        # cost per speed
        self.cost = 1
        self.honor = FLAGS.factor * self.cost
        self.last_r = 0
        # single uav total
        self.SUE_ENERGY = (self.cost * FLAGS.max_speed + self.honor)
        self.SUT_ENERGY = self.SUE_ENERGY * FLAGS.max_epoch
        self.dis_flag = False
        self.agent_index_for_greedy = 0
        # custom parameters for uav end---------------------------------------------------------------------------------
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,))
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,))
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(self.agents, self.world, self.poi_array, self.M)[0])
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def is_covered(self, pos):
        for uav in self.state:
            if self.get_distance(pos, uav) <= self.radius:
                return True
        return False

    def is_covered_for_greedy(self, pos, greedy_states_temp):
        for uav in greedy_states_temp:
            if self.get_distance(pos, uav) <= self.radius:
                return True
        return False

    def get_distance(self, a, b):
        delta_pos_x = a[0] - b[0]
        delta_pos_y = a[1] - b[1]
        dist = delta_pos_x ** 2 + delta_pos_y ** 2
        return dist

    def __set_matrix(self, x, y, matrix, value):
        assert x >= 0
        assert y < self.size
        _i = self.size - y - 1
        _j = x
        matrix[_i][_j] = value

    def __add_matrix(self, x, y, matrix, addition):
        assert x >= 0
        assert y < self.size
        _i = self.size - y - 1
        _j = x
        matrix[_i][_j] += addition

    def __get_matrix(self, x, y, matrix):
        assert x >= 0
        assert y < self.size
        _i = self.size - y - 1
        _j = x
        return matrix[_i][_j]

    def is_disconnected(self, state_current):
        state_current_temp = np.array(state_current).copy()
        dis_con = True
        graph = nx.Graph()
        graph.add_nodes_from(range(FLAGS.num_uav))
        for i in range(FLAGS.num_uav):
            loc_x = state_current_temp[i][0]
            loc_y = state_current_temp[i][1]
            if abs(loc_x) > FLAGS.map_constrain:
                loc_x = np.sign(loc_x) * FLAGS.map_constrain
            if abs(loc_y) > FLAGS.map_constrain:
                loc_y = np.sign(loc_y) * FLAGS.map_constrain
            state_current_temp[i][0] = loc_x
            state_current_temp[i][1] = loc_y
        for uav_i in range(FLAGS.num_uav):
            for uav_j in range(uav_i + 1, FLAGS.num_uav):
                connected_constraint = FLAGS.constrain ** 2
                if self.get_distance(state_current_temp[uav_i], state_current_temp[uav_j]) <= connected_constraint:
                    graph.add_edge(uav_j, uav_i)
        if nx.is_connected(graph):
            dis_con = False
        return dis_con

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        random_action_move_dis = []
        for i, agent in enumerate(self.agents):
            if FLAGS.greedy_action:
                greedy_action = self.greedy_algorithm(self.agent_index_for_greedy)
                if i == self.agent_index_for_greedy % len(self.agents):
                    action_n[i][1] = action_n[i][2] + greedy_action[0]
                    action_n[i][3] = action_n[i][4] + greedy_action[1]
                else:
                    action_n[i][1] = action_n[i][2]
                    action_n[i][3] = action_n[i][4]
            if FLAGS.random_action:
                random_action, move_dis_tem = self.random_action_algorithm()
                random_action_move_dis.append(move_dis_tem)
                action_n[i][1] = action_n[i][2] + random_action[0]
                action_n[i][3] = action_n[i][4] + random_action[1]
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # custom code for uav begin-------------------------------------------------------------------------------------
        # update state and map_state and energy
        self.agent_index_for_greedy += 1
        self.old_energy = self.energy.copy()
        state_current = []
        self.dis = 0
        self.over_map = 0
        for a in self.agents:
            loc_x = a.state.p_pos[0] * self.map_scale_rate
            loc_y = a.state.p_pos[1] * self.map_scale_rate
            location_tem = [loc_x, loc_y]
            state_current.append(location_tem)
            if abs(loc_x) > FLAGS.map_constrain or abs(loc_y) > FLAGS.map_constrain:
                self.over_map += 1
        if self.is_disconnected(state_current):
            self.dis_flag = True
            self.dis += 1
        else:
            self.dis_flag = False
        # custom code for uav end --------------------------------------------------------------------------------------
        # record observation for each agent
        obs_n = self._get_obs(self.agents)
        for agent in self.agents:
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        # custom code for uav begin-------------------------------------------------------------------------------------
        self.state = []
        for a in self.agents:
            location_tem = [a.state.p_pos[0] * self.map_scale_rate, a.state.p_pos[1] * self.map_scale_rate]
            self.state.append(location_tem)
        for agent_i, agent in enumerate(self.agents):
            move_distance = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1])) * 0.1 *\
                            FLAGS.map_scale_rate
            if FLAGS.greedy_action:
                move_distance = FLAGS.greedy_act_dis
            if FLAGS.random_action:
                move_distance = random_action_move_dis[agent_i]
            self.energy[agent_i] += self.cost * move_distance + self.honor

        delta_energy = np.sum(self.energy - self.old_energy) * 1.0 / (self.SUE_ENERGY * self.uav)
        self.delta = 0
        self.tmp = np.zeros([self.size, self.size], dtype=np.int8)

        for x in range(self.size):
            for y in range(self.size):
                cov = self.is_covered(self.PoI[x * self.size + y])
                if cov > 0:
                    if self.__get_matrix(x, y, self.tmp) != 1:
                        self.__add_matrix(x, y, self.final, 1)
                        self.delta += 1
                    self.__set_matrix(x, y, self.tmp, 1)

                self.__set_matrix(x, y, self.M,
                                  float(self.__get_matrix(x, y, self.final)) /
                                  FLAGS.max_epoch)
        x_sum = np.sum(self.M)
        x_square_sum = np.sum(self.M ** 2)
        self.jain_index = x_sum ** 2 / x_square_sum / self.size ** 2
        delta_coverage = self.delta * 1.0 / self.max_epoch
        reward_positive = delta_coverage * self.jain_index / delta_energy
        reward_positive_n = np.ones(self.uav) * reward_positive
        self.o_r = reward_positive
        # add positive reward
        reward_n = reward_n + reward_positive_n

        return obs_n, reward_n, done_n, info_n
        # custom code for uav end---------------------------------------------------------------------------------------

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        
        obs_n = self._get_obs(self.agents)

        # map reset begin-----------------------------------------------------------------------------------------------
        # energy reset
        self.energy = np.zeros(self.uav)
        self.M = np.zeros((self.size, self.size))
        self.MapState = np.zeros(self.size ** 2)
        self.final = np.zeros((self.size, self.size), dtype=np.int64)
        self.state = []
        self.tmp = np.zeros([self.size, self.size], dtype=np.int8)
        for a in self.agents:
            location_tem = [a.state.p_pos[0] * self.map_scale_rate, a.state.p_pos[1] * self.map_scale_rate]
            self.state.append(location_tem)
        # map reset end-------------------------------------------------------------------------------------------------
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agents):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agents, self.world, self.poi_array, self.M)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, self.dis_flag)

    def random_action_algorithm(self):
        # [a, b] (b-a)*random_sample() + a
        action_step_dis = FLAGS.greedy_act_dis / FLAGS.map_scale_rate
        action_result = -action_step_dis + action_step_dis * 2 * np.random.random((2, 1))
        move_dis = np.sqrt(np.square(action_result[0]) + np.square(action_result[1])) * FLAGS.map_scale_rate
        return action_result, move_dis

    def greedy_algorithm(self, act_agent_index):
        # todo calculate reward based on aciton space
        action_space_tem_len = 30
        action_step_dis = FLAGS.greedy_act_dis / FLAGS.map_scale_rate
        action_space_tem = []
        theta_tem = 0
        # action_space_tem.append([0, 0])
        for i in range(action_space_tem_len):
            theta_tem += (2 * np.pi / action_space_tem_len)
            action_space_tem.append([action_step_dis * np.cos(theta_tem), action_step_dis * np.sin(theta_tem)])
        state_greedy = []
        for a in self.agents:
            loc_x = a.state.p_pos[0] * self.map_scale_rate
            loc_y = a.state.p_pos[1] * self.map_scale_rate
            location_tem = [loc_x, loc_y]
            state_greedy.append(location_tem)
        max_reward_index = 0
        max_reward = -1e6
        act_i = act_agent_index % len(self.agents)
        for i in range(action_space_tem_len):
            state_greedy_tem = np.array(state_greedy).copy()
            map_coverage_score = self.M.copy()
            final_coverage_num = self.final.copy()
            state_greedy_tem[act_i][0] += (action_space_tem[i][0] * FLAGS.map_scale_rate)
            state_greedy_tem[act_i][1] += (action_space_tem[i][1] * FLAGS.map_scale_rate)
            reward_temp = self.greedy_calculate_reward(map_coverage_score, final_coverage_num, act_i,
                                                       state_greedy_tem, FLAGS.greedy_act_dis)
            if reward_temp > max_reward:
                max_reward = reward_temp
                max_reward_index = i
        return action_space_tem[max_reward_index]

    def greedy_calculate_reward(self, map_coverage_score, final_coverage_num, act_agent_index, agent_states,
                                action_step_dis):
        # todo calculate reward based on current states
        result_reward = 0
        tmp = np.zeros([self.size, self.size], dtype=np.int8)
        delta_num = 0
        delta_energy = self.cost * action_step_dis + self.honor
        for x in range(self.size):
            for y in range(self.size):
                cov = self.is_covered_for_greedy(self.PoI[x * self.size + y], agent_states)
                if cov > 0:
                    if self.__get_matrix(x, y, tmp) != 1:
                        self.__add_matrix(x, y, final_coverage_num, 1)
                        delta_num += 1
                    self.__set_matrix(x, y, tmp, 1)

                self.__set_matrix(x, y, map_coverage_score, float(self.__get_matrix(x, y, final_coverage_num))
                                  / FLAGS.max_epoch)
        x_sum = np.sum(map_coverage_score)
        x_square_sum = np.sum(map_coverage_score ** 2)
        jain_index_tem = x_sum ** 2 / x_square_sum / self.size ** 2
        delta_coverage = delta_num * 1.0 / self.max_epoch
        reward_positive = delta_coverage * jain_index_tem / delta_energy
        result_reward += reward_positive
        if self.is_disconnected(agent_states):
            result_reward -= FLAGS.penalty_disconnected
        def bound(x):
            if x < 4.5:
                return 0
            if x < 5:
                return (x - 4.5) * FLAGS.penalty
            return min(np.exp(2 * x - 2), FLAGS.penalty)
        result_reward -= bound(agent_states[act_agent_index][0])
        result_reward -= bound(agent_states[act_agent_index][1])
        return result_reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, spaces.MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= FLAGS.action_sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def _render(self, mode='human', close=True):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        if close:
            # close any existic renderers
            for i,viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    # custom return to global train------------------------------------------------------------------------------------
    def _get_energy(self):
        return np.mean(self.energy.copy() / self.SUT_ENERGY)

    def _get_energy_origin(self):
        return np.mean(self.energy.copy())

    def _get_jain_index(self):
        return self.jain_index

    def _get_delta_c(self):
        return self.delta

    def _get_dis(self):
        return self.dis

    def _get_over_map(self):
        return self.over_map

    def _get_original_r(self):
        return self.o_r

    def _get_aver_cover(self):
        total_cover = np.sum(self.M)
        return total_cover / (self.size ** 2)

    def _get_state(self):
        tmp = []
        for state in self.state:
            tmp.append(state[0])
            tmp.append(state[1])
        return tmp

    def _get_original_r(self):
        return self.o_r


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def _step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def _reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def _render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
