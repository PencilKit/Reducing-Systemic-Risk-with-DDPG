from gym.envs.registration import register

register(
    id='network_env-v0',
    entry_point="network_gym.envs:network_env",
)