from gym import envs

all_envs = envs.registry.all()
print(type(all_envs))
for env in all_envs:
    print(type(env))
    print(env)
