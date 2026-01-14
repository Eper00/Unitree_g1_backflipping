import gymnasium as gym
import gymnasium_env
env = gym.make('gymnasium_env/UnitreeG1-v0', render_mode='human')
env.reset()
env.render()