import subprocess
import os
import sys
sys.path.append("/home/mxxmhh/mxxhcm/code/maddpg/")

from multiagent.uav.flag import FLAGS


path = "./tmp/num_uav_" + str(FLAGS.num_uav) + "_radius_"+str(FLAGS.radius)
"_factor_"+ str(FLAGS.factor) + "_constrain_"+str(FLAGS.constrain)
print(path)
for i in range(99, 399, 100):
    subprocess_command = ['python', 'test.py', '--episode', str(i), '--load-dir', path]
    os_command = 'python test.py --episode 299 --load-dir ./tmp/num_uav_6_radius_1.0'

    # print(os_command)
    os.system(os_command)
    # subprocess.run(command)
