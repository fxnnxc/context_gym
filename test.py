from os import system
import gym 
from context_gym import LunarLanderWrapper, make_env


pairs = [
    ("LunarLander-v2", LunarLanderWrapper, ['gravity'])
]

for env_id, wrapper, params in pairs:
    print("----------------")
    print(env_id, wrapper)
    print("----------------")
    envs = gym.vector.SyncVectorEnv([make_env("LunarLander-v2", 0, 0, False,"test", wrapper=LunarLanderWrapper, system_params=params) for i in range(2)])
    envs.reset()
    for i in range(1000):
        s, r, d, info = envs.step(envs.action_space.sample())
        for i in range(len(envs.envs)):
            if d[i]:
                envs.envs[i].set_context(envs.envs[i].sample_context())
                print(info['context'])
            
