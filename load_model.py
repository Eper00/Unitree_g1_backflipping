import torch
import gymnasium as gym
import gymnasium_env
from ppo import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    return gym.make("gymnasium_env/UnitreeG1-v0", render_mode="human")

envs = gym.vector.SyncVectorEnv([make_env])

agent = Agent(envs).to(device)

model_path = "/home/tomi/Unitree_g1_backflipping/runs/gymnasium_env/UnitreeG1-v0__ppo__1__1768594200/ppo.cleanrl_model"
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()

# 🔹 RESET (kezdeti pozíció!)
obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32).to(device)
actions=[]
done = False
while not done:
    with torch.no_grad():
        action ,_,_,_= agent.get_action_and_value(obs)
        actions.append(action)
    
   
    obs, reward, terminated, truncated, _ = envs.step(action)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)

    done = terminated[0] or truncated[0]

envs.close()
env = gym.make("gymnasium_env/UnitreeG1-v0")
obs, info = env.reset()
for i in actions:
    with torch.no_grad():
        env.step(i.numpy())
env.render()
