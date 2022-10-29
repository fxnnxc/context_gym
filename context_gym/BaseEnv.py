from typing_extensions import Self
import numpy as np 
import gym 

class ContextEnvironment(gym.Wrapper):
    def __init__(self, env, history_len, clip_system_params, normalize_system_params):
        super().__init__(env)
        self.normalize_system_params = normalize_system_params
        self.clip_system_params = clip_system_params
        self.obs_size = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)
        self.act_size = self.action_space.shape if  hasattr(env.action_space, "shape") else tuple(1,)
        observation_space = {"state" : env.observation_space,
                            "history_obs" : gym.spaces.Box(-np.inf, np.inf, (history_len, *self.obs_size)),
                            'history_act' : gym.spaces.Box(-np.inf, np.inf, (history_len, *self.act_size)),
                            "context" : gym.spaces.Box(-np.inf, np.inf, (len(self.system_params),)),
             }
    
        self.observation_space = gym.spaces.Dict(observation_space)
        self.history_len = history_len
        self.prev_state = None 
        self.history_obs = np.zeros((history_len, *self.obs_size))
        self.history_act = np.zeros((history_len, *self.act_size)).reshape(self.history_len, -1)
    
    def sample_context(self):
        # generate random context
        method = self.sampling_config['sample']
        params = self.sampling_config['params']
        
        if self.clip_system_params:
            INTERVALS = self.ALL_PARAMS
            context = {k : np.clip(method(v), INTERVALS[k][0], INTERVALS[k][1])    for k,v in params.items()}
        else:
            context = {k : method(v) for k,v in params.items()} 

        if self.normalize_system_params:
            INTERVALS = params
            context = {k : (v - INTERVALS[k][0])/(INTERVALS[k][1] - INTERVALS[k][0])    for k,v in context.items()}
            print(context)

        return context
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.history_obs = np.concatenate([self.history_obs[1:],  self.prev_state.reshape(1,-1)], axis=0)
        self.history_act = np.concatenate([self.history_act[1:],  np.array(action).reshape(1,-1)], axis=0)
        dict_state = {
            "state" : next_state,
            "history_obs" : self.history_obs.copy(),
            "history_act" : self.history_act.copy(),
            "context" : np.array([v for k,v in self.get_context().items()])
        }
        self.prev_state = next_state 
        return dict_state, reward, done, info
    def reset(self):
        state =  super().reset()
        self.prev_state = state 
        self.history_obs = np.zeros((self.history_len, *self.obs_size))
        self.history_act = np.zeros((self.history_len, *self.act_size)).reshape(self.history_len, -1)
        state = {
            "state" : state,
            "history_obs" : self.history_obs.copy(),
            "history_act" : self.history_act.copy(),
            "context" : np.array([v for k,v in self.get_context().items()])
        }
        return state 
