import matplotlib.pyplot as plt
import os
import sys
cwd = os.getcwd()
path = cwd + "/../"
sys.path.append(path)

from multiagent.uav.flag import FLAGS
import numpy as np


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def draw(i, path, energy, route, actions, ob_, sqrt_, r_, discon_, over_map, final_steps, Run = False):
    mkdir(path)
    label = 'epoch:' + str(FLAGS.max_epoch) + '\nUAV: ' + str(FLAGS.num_uav) + '\n map size: ' + str(FLAGS.size_map) + '\n sensing range:' + str(FLAGS.radius) \
            + '\n constraint:' + str(FLAGS.constrain)

    Fig = plt.figure(figsize=(18, 10))  # Create a `figure' instance

    Ax = Fig.add_subplot(321)
    plt.xlabel('No. of epochs')
    plt.ylabel('Average attained coverage')
    Ax.plot(range(final_steps), ob_)

    # #
    Bx = Fig.add_subplot(322)
    plt.xlabel('No. of epochs')
    plt.ylabel('Jain\'s fairness index')
    Bx.plot(range(final_steps), sqrt_)

    # #
    Cx = Fig.add_subplot(323)
    plt.xlabel('No. of epochs')
    plt.ylabel('Accumulated reward')
    Cx.plot(range(final_steps), r_)

    # #
    Dx = Fig.add_subplot(324)
    plt.xlabel('No. of epochs')
    plt.ylabel('Accumulated times \nof disconnection')
    Dx.plot(range(final_steps), discon_, color='blue')

    Gx = Fig.add_subplot(326)
    plt.xlabel('No. of epochs')
    plt.ylabel('Accumulated times \nto fly outside the map')
    line_ob, = Gx.plot(range(final_steps), over_map, color='green')
    plt.legend([line_ob, ], [label, ])

    Hx = Fig.add_subplot(325)
    plt.xlabel('No. of epochs')
    plt.ylabel('Energy consumption')
    Hx.plot(range(final_steps), energy, color='green')

    Fig.subplots_adjust(hspace=0.4)
    Fig.savefig(path + '/pic_' + str(i) + '.png')
    plt.close()


def drawTest(i, path, energy, coverage, jainindex, r_, discon_, over_map, final_steps, BL_coverage, BL_jain, BL_loss, energy_efficiency, Run = False):
    mkdir(path)
    label = 'epoch:' + str(FLAGS.max_epoch) + '\nUAV: ' + str(FLAGS.num_uav) + '\n map size: ' + str(FLAGS.size_map) + '\n sensing range:' + str(FLAGS.radius) \
            + '\n constraint:' + str(FLAGS.constrain) + '\n average energy efficiency:' + str(np.mean(energy_efficiency)) \
            + '\n max energy efficiency:' + str(np.max(energy_efficiency))

    Fig = plt.figure(figsize=(18, 10))  # Create a `figure' instance

    Ax = Fig.add_subplot(421)
    plt.xlabel('No. of episodes')
    plt.ylabel('Average attained coverage')
    Ax.plot(range(final_steps), coverage)
    Ax.plot([BL_coverage] * final_steps)

    # #
    Bx = Fig.add_subplot(422)
    plt.xlabel('No. of episodes')
    plt.ylabel('Jain\'s fairness index')
    Bx.plot(range(final_steps), jainindex)
    Bx.plot([BL_jain] * final_steps)

    # #
    Cx = Fig.add_subplot(423)
    plt.xlabel('No. of episodes')
    plt.ylabel('Instantaneous reward')
    Cx.plot(range(final_steps), r_)

    # #
    Dx = Fig.add_subplot(424)
    plt.xlabel('No. of episodes')
    plt.ylabel('Instantaneous times \nof disconnection')
    Dx.plot(range(final_steps), discon_, color='blue')
    Dx.plot([BL_loss] * final_steps)

    Gx = Fig.add_subplot(426)
    plt.xlabel('No. of episodes')
    plt.ylabel('Accumulated times \nto fly outside the map')
    line_ob, = Gx.plot(range(final_steps), over_map, color='green')
    plt.legend([line_ob, ], [label, ])

    Hx = Fig.add_subplot(425)
    plt.xlabel('No. of episodes')
    plt.ylabel('Average energy consumption')
    Hx.plot(range(final_steps), energy, color='green')

    Hx = Fig.add_subplot(427)
    plt.xlabel('No. of episodes')
    plt.ylabel('Energy efficiency')
    Hx.plot(range(final_steps), energy_efficiency, color='magenta')

    Fig.subplots_adjust(hspace=0.4)
    Fig.savefig(path + '/pic_' + str(i) + '.png')
    plt.close()


def draw_single_episode(path, episode_number, efficiency,
                        coverage, fairness, energy, disconnect, over_map, reward):
    mkdir(path)
    steps = len(efficiency)
    plt.figure(figsize=(40, 20))
    plt.subplot(4, 2, 1)
    plt.xlabel("No. of step")
    plt.ylabel("Energy efficiency")
    plt.plot(range(steps), efficiency, color='b')

    plt.subplot(4, 2, 2)
    plt.xlabel("No. of step")
    plt.ylabel("Coverage")
    plt.plot(range(steps), coverage, color='g')

    plt.subplot(4, 2, 3)
    plt.xlabel("No. of step")
    plt.ylabel("Fairness")
    plt.plot(range(steps), fairness, color='r')

    plt.subplot(4, 2, 4)
    plt.xlabel("No. of step")
    plt.ylabel("Energy")
    plt.plot(range(steps), energy, color='c')

    plt.subplot(4, 2, 5)
    plt.xlabel("No. of step")
    plt.ylabel("Disconnect")
    plt.plot(range(steps), disconnect, color='m')

    plt.subplot(4, 2, 6)
    plt.xlabel("No. of step")
    plt.ylabel("Over map counter")
    plt.plot(range(steps), over_map, color='y')

    plt.subplot(4, 2, 7)
    plt.xlabel("No. of step")
    plt.ylabel("Reward")
    plt.plot(range(steps), over_map, color='w')

    plt.savefig(path + "episode_" + str(episode_number) + '_info.png')
    plt.close()


def draw_episodes(i, path, coverage, j_index, energy, A_discon, A_over_map, A_efficiency, A_reward, final_steps):
    mkdir(path)
    steps = final_steps
    plt.figure(figsize=(18, 10))

    plt.subplot(4, 2, 1)
    plt.xlabel("No. of step")
    plt.ylabel("Fairness")
    plt.plot(range(steps), j_index, color='r')

    plt.subplot(4, 2, 2)
    plt.xlabel("No. of step")
    plt.ylabel("Coverage")
    plt.plot(range(steps), coverage, color='g')

    plt.subplot(4, 2, 3)
    plt.xlabel("No. of step")
    plt.ylabel("Energy")
    plt.plot(range(steps), energy, color='c')

    plt.subplot(4, 2, 4)
    plt.xlabel("No. of step")
    plt.ylabel("Energy efficiency")
    plt.plot(range(steps), A_efficiency, color='b')

    plt.subplot(4, 2, 5)
    plt.xlabel("No. of step")
    plt.ylabel("Disconnect")
    plt.plot(range(steps), A_discon, color='m')

    plt.subplot(4, 2, 6)
    plt.xlabel("No. of step")
    plt.ylabel("Over map counter")
    plt.plot(range(steps), A_over_map, color='y')

    plt.subplot(4, 2, 7)
    plt.xlabel("No. of step")
    plt.ylabel("Reward")
    plt.plot(range(steps), A_reward, color='w')

    plt.savefig(path
                + "episodes_" + str(i) + '.png')
    plt.close()
