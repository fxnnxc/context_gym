import gym 
from context_gym.BaseEnv import ContextEnvironment
from context_gym.LunarLander import LunarLanderContinuousWrapper
from context_gym.CartPoleContinuous import CartPoleContinuousWrapper
from context_gym.Pendulum import PendulumWrapper 
from context_gym.Acrobot import AcrobotWrapper
from context_gym.HalfCheetah import HalfCheetahWrapper
from context_gym.Ant import AntWrapper
from context_gym.Hopper import HopperWrapper
import numpy as np 


def make_env(env_id, seed, idx, capture_video, run_name, wrapper=None, system_params=None, history_len=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if wrapper is not None:
            assert system_params is not None and history_len is not None 
            env = wrapper(env, system_params, history_len)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # clip the environment when the agent is ppo
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

