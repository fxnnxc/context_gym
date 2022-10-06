
import gym 
import numpy as np 
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

SAMPLING_NORMAL = {
    "sample" : lambda v : np.random.normal(v[0], v[1]),
    "params":{
        'gravity' : [9.8, 1.0],  # toward downside 
        'length' : [1.0, 0.25],
        'mass' : [1.0, 0.25],
        'max_speed' : [8, 1],
        'max_torque' : [2.0, 1.0]
    }
}
SAMPLING_UNIFORM = {
    "sample" : lambda v : np.random.uniform(v[0], v[1]),
    "params":{
        'gravity' : [1.0, 12.0],
        'length' : [0.1, 2.0],
        'mass' : [0.1, 2.0],
        'max_speed' : [5, 15],
        'max_torque' : [1.0, 3.0]
    }
}

class PendulumWrapper(gym.Wrapper):
    
    ALL_PARAMS  = {
        'gravity' : [1.0, 12.0],
        'length' : [0.1, 2.0],
        'mass' : [0.1, 2.0],
        'max_speed' : [5, 15],
        'max_torque' : [1.0, 3.0]
    }
    
    def __init__(self, env, system_params, history_len, sampling_config=SAMPLING_UNIFORM, clip_system_params=False):
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
        assert len(set(self.system_params) - set(PendulumWrapper.ALL_PARAMS.keys())) == 0
        self.sampling_config = sampling_config
        self.clip_system_params =clip_system_params 
        
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
        
        if self.clip_system_params:
            INTERVALS = PendulumWrapper.ALL_PARAMS
            context = {k : np.clip(method(v), INTERVALS[k][0], INTERVALS[k][1])    for k,v in params.items()}
        else:
            context = {k : method(v) for k,v in params.items()} 

        return context
    
    def set_context(self, context):
        # define how to set environment variables 
        if "gravity" in self.system_params:
            self.env.unwrapped.g = context['gravity']
        if 'length' in self.system_params:
            self.env.unwrapped.l = context['length']
        if 'mass' in self.system_params:
            self.env.unwrapped.m = context['mass']
        if 'max_speed' in self.system_params:
            self.env.unwrapped.max_speed = context['max_speed']
        if 'max_torque' in self.system_params:
            self.env.unwrapped.max_torque = context['max_torque']
        
    def get_context(self):
        context = {} 
        if "gravity" in self.system_params:
            context['gravity'] = self.env.unwrapped.g
        if 'length' in self.system_params:
            context['length'] = self.env.unwrapped.l
        if 'mass' in self.system_params:
            context['mass'] = self.env.unwrapped.m
        if 'max_speed' in self.system_params:
            context['max_speed'] = self.env.unwrapped.max_speed
        if 'max_torque' in self.system_params:
            context['max_torque'] = self.env.unwrapped.max_torque
        return context 
    
    
if __name__ == "__main__":
    env = PendulumWrapper(gym.make("Pendulum-v1"), ['gravity', 'length', 'mass', 'max_speed', 'max_torque'], 3,)
    
    for i in range(100):
        done = False         
        print(env.get_context())
        context = env.sample_context()
        env.set_context(context)
        env.reset()
        count= 0 
        while not done:
            action = env.action_space.sample()
            env.tau = None
            ns, r, done, info = env.step(action)
            env.render()
            count +=1 
        print(count)
            

        
            