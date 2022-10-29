
import gym 
import numpy as np 
from context_gym import ContextEnvironment



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

class PendulumWrapper(ContextEnvironment):
    
    def __init__(self, env, history_len, clip_system_params, normalize_system_params, sampling_config=SAMPLING_UNIFORM):
        self.ALL_PARAMS  = {
                'gravity' : [1.0, 12.0],
                'length' : [0.1, 2.0],
                'mass' : [0.1, 2.0],
                'max_speed' : [5, 15],
                'max_torque' : [1.0, 3.0]
            }
            
        self.system_params = list(sampling_config['params'].keys())
        self.sampling_config = sampling_config
        super().__init__(env, history_len, clip_system_params, normalize_system_params)
        assert len(set(self.system_params) - set(self.ALL_PARAMS.keys())) == 0

    
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
            

        
            