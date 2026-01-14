from gymnasium.envs.registration import register

register(
    id="gymnasium_env/UnitreeG1-v0",
    entry_point="gymnasium_env.envs:UnitreeG1Env",
)
