import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import networkx as nx
from PIL import Image, ImageDraw
from multiagent.uav.flag import FLAGS


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = FLAGS.num_uav
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.max_speed = FLAGS.max_speed / FLAGS.map_scale_rate / 0.1
            agent.size = FLAGS.radius / FLAGS.map_scale_rate
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(0, 2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        rew = 0
        # agents are penalized for exiting the screen
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def get_distance(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist

    def calculate_positive_reward(self, agent, world):
        image_size = 20
        image = Image.new('1', (image_size, image_size), 255)
        draw = ImageDraw.Draw(image)
        for a in world.agents:
            begin_point_x = (a.state.p_pos[0] - a.size) * 10
            begin_point_y = (a.state.p_pos[0] - a.size) * 10
            end_point_x = (a.state.p_pos[1] - a.size) * 10
            end_point_y = (a.state.p_pos[1] - a.size) * 10
            draw.ellipse((begin_point_x, begin_point_y, end_point_x, end_point_y), fill=0)
        coverage_size = 0
        for i in range(image_size):
            for j in range(image_size):
                if image.getpixel((i, j))[0] == 0:
                    coverage_size += 1
        return coverage_size

    def reward(self, agent, world, dis_flag):
        rew = 0
        # agent are penalized for exiting the screen
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.:
                return (x - 0.9) * FLAGS.penalty
            return min(np.exp(2 * x - 2), FLAGS.penalty)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        # agent disconnected from the uav network
        if dis_flag:
            rew -= FLAGS.penalty_disconnected
        return rew

    def get_matrix(self, x, y, matrix):
        assert x >= 0
        assert y < FLAGS.size_map
        _i = FLAGS.size_map - y - 1
        _j = x
        return matrix[_i][_j]

    def observation(self, m, map, agents_positions, map_size):
        # channel 1: m
        # channel 2: map
        # channel 3: uav pos
        # map
        positions = np.zeros((map_size, map_size, 1))
        for pos_x, pos_y in agents_positions:
            positions[int(np.floor(pos_x))][int(np.floor(pos_y))][0] += 1
        obs = np.concatenate((np.expand_dims(map, 2), np.expand_dims(m, 2), positions), 2)
        # env_information = np.concatenate((loc, np.reshape(m, (FLAGS.size_map*FLAGS.size_map, 1))), 1)
        # env_list = env_information.tolist()
        # comm = []
        # pos = []
        # obs_n = []
        # for agent in agents:
        #     comm.append(agent.state.c)
        #     pos.append(agent.state.p_pos)
        # pos_array = np.asarray(pos)
        # info = np.concatenate(env_list)
        # info_list = info.tolist()
        # for i, agent in enumerate(agents):
        #     pos = pos_array - pos_array[i]
        #     pos[i] = agent.state.p_pos
        #     obs_n.append(np.concatenate([agent.state.p_vel]+pos.tolist()+[info_list]+comm))
        return obs