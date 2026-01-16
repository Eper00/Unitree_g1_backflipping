import gymnasium as gym
import gymnasium_env

from ppo import train, Args

if __name__ == "__main__":
    env = gym.make("gymnasium_env/UnitreeG1-v0")
    obs, info = env.reset()

    args = Args(
        env_id="gymnasium_env/UnitreeG1-v0",
        total_timesteps=1_000_000,
        num_envs=1,
        cuda=True,
        save_model=True,
        
    )


    train(args)
    #env.render()
