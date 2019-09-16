import retro
import time
import numpy as np

env = retro.make(game='Airstriker-Genesis')
state = env.reset()
    
while True:
    state, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        time.sleep(0.3)
        env.reset()

