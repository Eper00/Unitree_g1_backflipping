import gymnasium as gym
import gymnasium_env
import numpy as nps
import numpy as np



env = gym.make('gymnasium_env/UnitreeG1-v0')
obs, info = env.reset()

for i in range (1000):
    action = np.ones(env.unwrapped.model.nu) * i/1000
    obs, reward, terminated, truncated, info = env.step(action)

env.render()
