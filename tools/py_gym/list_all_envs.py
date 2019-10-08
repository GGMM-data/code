from gym import envs

all_envs = envs.registry.all()
print(type(all_envs))
count = 0
for env in all_envs:
    if count > 10:
        break
    print(type(env))
    print(env)
    print(type(env.entry_point))
    print(env.entry_point)
    print(env.entry_point.split(':')[0].split('.')[-1])
    count += 1
