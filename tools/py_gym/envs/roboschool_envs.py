import roboschool
import gym

count = 0
for env in gym.envs.registry.all():
    # print(env)
    if env.id.startswith('Roboschool'):
        print("\n".join([env.id, "==================="]))
        count += 1

print(count)
