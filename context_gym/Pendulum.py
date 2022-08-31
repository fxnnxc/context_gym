
import gym 
import numpy as np 
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

SAMPLING_NORMAL = {
    "sample" : lambda mean, std : np.random.normal(mean, std),
    "params":{
        'gravity' : [9.8, 1.0],  # toward downside 
        'length' : [1.0, 0.25],
        'mass' : [1.0, 0.25],
        'max_speed' : [8, 1],
        'max_torque' : [2.0, 1.0]
    }
}
SAMPLING_UNIFORM = {
    "sample" : lambda left, right : np.random.uniform(left, right),
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
    
    def __init__(self, env, system_params, history_len, sampling_config=SAMPLING_UNIFORM):
        super().__init__(env)
        size = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)
        self.observation_space = gym.spaces.Dict(
            {"state" : env.observation_space,
             "history" : env.observation_space.__class__(-np.inf, np.inf, (history_len, *size)),
             "context" : gym.spaces.Box(-np.inf, np.inf, (len(system_params), )) 
             }
        )
        self.history = np.zeros((history_len, *size))
        self.system_params = system_params 
        assert len(set(self.system_params) - set(PendulumWrapper.ALL_PARAMS.keys())) == 0
        self.sampling_config = sampling_config
        
        
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        self.history = np.concatenate([self.history[1:],  next_state.reshape(1,-1)], axis=0)
        next_state = {
            "state" : next_state,
            "history" : self.history.copy(),
            "context" : np.array([v for k,v in self.get_context().items()])
        }
        return next_state, reward, done, info
    def reset(self):
        state =  super().reset()
        self.history = np.concatenate([self.history[1:],  state.reshape(1,-1)], axis=0)
        state = {
            "state" : state,
            "history" : self.history.copy(),
            "context" : np.array([v for k,v in self.get_context().items()])
        }
        return state 
    
    def sample_context(self):
        # generate random context
        method = self.sampling_config['sample']
        params = self.sampling_config['params']
        
        INTERVALS = PendulumWrapper.ALL_PARAMS
        context = {k : np.clip(method(v[0], v[1]), INTERVALS[k][0], INTERVALS[k][1])    for k,v in params.items()} 
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
            

        
            