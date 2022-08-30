from stable_baselines3.common.buffers import DictReplayBuffer
import gym 
from context_gym import LunarLanderWrapper, make_env
import numpy as np 

pairs = [
    ("LunarLanderContinuous-v2", LunarLanderWrapper, ['gravity_x', 'gravity_y'])
]

history_len = 3
for env_id, wrapper, params in pairs:
    print("----------------")
    print(env_id, wrapper)
    print("----------------")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, False,"test", 
                                              wrapper=wrapper, 
                                              system_params=params, 
                                              history_len=history_len)])
    print(envs.single_action_space)
    print(envs.single_observation_space)
    rb = DictReplayBuffer(
            1000,
            envs.single_observation_space,
            envs.single_action_space,
            "cpu",
            handle_timeout_termination=True,
    )
    obs = envs.reset()
    for i in range(1000):
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_obs, rewards, dones, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                # real_next_obs[idx] = infos[idx]["terminal_observation"]
                envs.envs[idx].set_context(envs.envs[idx].sample_context())
                data = rb.sample(3)
                print(data.observations)

        rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        obs = next_obs

