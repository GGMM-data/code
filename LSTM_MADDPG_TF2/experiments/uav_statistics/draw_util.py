import matplotlib.pyplot as plt
import os
from LSTM_MADDPG_TF2.multiagent.uav.flag import FLAGS
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

    # #
    # route = np.array(route)
    #
    # for uav_i in range(FLAGS.num_uav):
    #     fig_tem = plt.figure(figsize=(10, 10))
    #     ax = fig_tem.add_subplot(111)
    #     x = route[uav_i][0]
    #     y = route[uav_i][1]
    #     x_ = route[FLAGS.num_uav * FLAGS.max_epoch - (FLAGS.num_uav - uav_i)][0]
    #     y_ = route[FLAGS.num_uav * FLAGS.max_epoch - (FLAGS.num_uav - uav_i)][1]
    #     l1 = ax.scatter(route[uav_i::FLAGS.num_uav, 0], route[uav_i::FLAGS.num_uav, 1], color='b', marker='o')
    #     plt.plot(route[uav_i::FLAGS.num_uav, 0], route[uav_i::FLAGS.num_uav, 1])
    #     l2 = ax.scatter(x, y, c='y', marker='o')
    #     l3 = ax.scatter(x_, y_, c='r', marker='o')
    #     plt.legend(handles = [l1, l2, l3], labels = ['track point', 'starting point', 'end point'], loc = 'best')
    #     plt.ylim(ymin=0, ymax=FLAGS.size_map)
    #     plt.xlim(xmin=0, xmax=FLAGS.size_map)
    #     fig_tem.savefig(path + '/pic_' + str(i) + ':UAV_' + str(uav_i + 1) +'.png')
    #     plt.close()


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


def draw_episode(i, path, coverage, j_index, A_reward, A_discon, A_over_map, loss, final_steps):
    mkdir(path)
    label = 'epoch:' + str(FLAGS.max_epoch) + '\nUAV: ' + str(FLAGS.num_uav) + '\n map size: ' + str(
        FLAGS.size_map) + '\n sensing range:' + str(FLAGS.radius) \
            + '\n constraint:' + str(FLAGS.constrain)

    with open("record.file"+"5", "a+") as file:
        record = str(i)+"\n"
        file.write(record)
        record = str(coverage) + "\n" + str(j_index) + "\n" + str(A_reward) + "\n"
        file.write(record)

    plt.figure(1,  figsize=(8, 6))  # Create a `figure' instance

    label = ['average cover score', 'fairness index']
    plt.xlabel('No. of time steps')
    plt.ylabel('Average attained coverage')
    plt.plot(range(final_steps), coverage, 'b-.', range(final_steps), j_index, 'r-')
    plt.legend(label)
    plt.savefig(path + "/episode_" + str(i) + "_cover_fair.png")
    plt.close()

    # #
    plt.figure(2, figsize=(8, 6))  # Create a `figure' instanc
    plt.xlabel('No. of episodes')
    plt.ylabel('over counter')
    plt.plot(range(final_steps), A_over_map)
    plt.savefig(path + "/episode_" + str(i) + "_over.png")
    plt.close()

    plt.figure(3, figsize=(8, 6))  # Create a `figure' instance
    plt.xlabel('No. of time steps')
    plt.ylabel('disconnect')
    plt.plot(range(final_steps), A_discon)
    plt.savefig(path + "/episode_" + str(i) + "_disconnect.png")
    plt.close()


    # #
    plt.figure(4, figsize=(8, 6))  # Create a `figure' instance
    plt.xlabel('No. of time steps')
    plt.ylabel('Accumulated reward')
    plt.plot(range(final_steps), A_reward)
    plt.savefig(path + "/episode_" + str(i) + "_reward.png")
    plt.close()

    #plt.show()
    # # #
    # Dx = Fig.add_subplot(324)
    # plt.xlabel('No. of episodes')
    # plt.ylabel('Instantaneous times \nof disconnection')
    # Dx.plot(range(final_steps), A_discon, color='blue')
    #
    # Gx = Fig.add_subplot(325)
    # plt.xlabel('No. of episodes')
    # plt.ylabel('Instantaneous times \nto fly outside the map')
    # line_ob, = Gx.plot(range(final_steps), A_over_map, color='green')
    # plt.legend([line_ob, ], [label, ])
    #
    # # Hx = Fig.add_subplot(325)
    # # plt.xlabel('No. of episodes')
    # # plt.ylabel('loss')
    # # # print('len loss:', len(loss), ' step:', final_steps)
    # # Hx.plot(range(final_steps), loss, color='green')
    #
    # Fig.subplots_adjust(hspace=0.4)
    # Fig.savefig(path + '/episode_' + str(i) + '.png')
    # plt.close()
