# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

import gym 
import numpy as np 

SAMPLING_NORMAL = {
    "sample" : lambda v : np.random.normal(v[0], v[1]),
    "params":{
        'LINK_MASS_1' : [1.0, 0.2],
        'LINK_MASS_2' : [1.0, 0.2],
        'LINK_LENGTH_1' : [1.0, 0.2],
        'LINK_LENGTH_2' : [1.0, 0.2],
    }
}

SAMPLING_UNIFORM = {
    "sample" : lambda v : np.random.uniform(v[0], v[1]),
    "params":{
        'LINK_MASS_1' : [0.1, 2.5],
        'LINK_MASS_2' : [0.1, 2.5],
        'LINK_LENGTH_1' : [0.3, 2.5],
        'LINK_LENGTH_2' : [0.3, 2.5],
    }
}

class AcrobotWrapper(gym.Wrapper):
    
    ALL_PARAMS  = {
        'LINK_MASS_1' : [0.1, 2.5],
        'LINK_MASS_2' : [0.1, 2.5],
        'LINK_LENGTH_1' : [0.3, 2.5],
        'LINK_LENGTH_2' : [0.3, 2.5],
    }
    
    def __init__(self, env, system_params, history_len, sampling_config=SAMPLING_UNIFORM):
        super().__init__(env)
        self.size = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)
        self.observation_space = gym.spaces.Dict(
            {"state" : env.observation_space,
             "history_obs" : env.observation_space.__class__(-np.inf, np.inf, (history_len, *self.size)),
             "history_act" : gym.spaces.Box(-np.inf, np.inf, (history_len, 1)),
             "context" : gym.spaces.Box(-np.inf, np.inf, (len(system_params), ))
             }
        )
        self.history_len = history_len
        self.prev_state = None 
        self.history_obs = np.zeros((history_len, *self.size))
        self.history_act = np.zeros((history_len, 1))
        self.system_params = system_params 
        assert len(set(self.system_params) - set(AcrobotWrapper.ALL_PARAMS.keys())) == 0
        self.sampling_config = sampling_config

        
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        self.history_obs = np.concatenate([self.history_obs[1:],  self.prev_state.reshape(1,-1)], axis=0)
        self.history_act = np.concatenate([self.history_act[1:],  np.array([action]).reshape(1,-1)], axis=0)
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
        self.history_obs = np.zeros((self.history_len, *self.size))
        self.history_act = np.zeros((self.history_len, 1))
        state = {
            "state" : state,
            "history_obs" : self.history_obs.copy(),
            "history_act" : self.history_act.copy(),
            "context" : np.array([v for k,v in self.get_context().items()])
        }
        return state 
    
    def sample_context(self):
        # generate random context
        method = self.sampling_config['sample']
        params = self.sampling_config['params']
        
        INTERVALS = AcrobotWrapper.ALL_PARAMS
        context = {k : np.clip(method(v), INTERVALS[k][0], INTERVALS[k][1]) for k,v in params.items()} 
        return context
    
    def set_context(self, context):
        # define how to set environment variables 
        if "LINK_MASS_1" in self.system_params:
            self.env.unwrapped.LINK_MASS_1 = context['LINK_MASS_1']
        if "LINK_MASS_2" in self.system_params:
            self.env.unwrapped.LINK_MASS_2 = context['LINK_MASS_2']
        if "LINK_LENGTH_1" in self.system_params:
            self.env.unwrapped.LINK_LENGTH_1 = context['LINK_LENGTH_1']
        if "LINK_LENGTH_2" in self.system_params:
            self.env.unwrapped.LINK_LENGTH_2 = context['LINK_LENGTH_2']
        
    def get_context(self):
        context = {} 
        if "LINK_MASS_1" in self.system_params:
            context['LINK_MASS_1'] = self.env.unwrapped.LINK_MASS_1
        if "LINK_MASS_2" in self.system_params:
            context['LINK_MASS_2'] = self.env.unwrapped.LINK_MASS_2
        if "LINK_LENGTH_1" in self.system_params:
            context['LINK_LENGTH_1'] = self.env.unwrapped.LINK_LENGTH_1
        if "LINK_LENGTH_2" in self.system_params:
            context['LINK_LENGTH_2'] = self.env.unwrapped.LINK_LENGTH_2
        return context 
    
if __name__ == "__main__":
    env = AcrobotWrapper(gym.make("Acrobot-v1"), ['LINK_MASS_1', 'LINK_MASS_2', 'LINK_LENGTH_1', 'LINK_LENGTH_2'], 3, )
    
    for i in range(100):
        done = False         
        env.reset()
        context = env.sample_context()
        env.set_context(context)
        print(env.get_context())

        count= 0 
        while not done:        
            action = env.action_space.sample()
            ns, r, done, info = env.step(action)
            # env.render()
            count +=1 
        print(count)
            

        
            