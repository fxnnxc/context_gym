# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

import gym 
import numpy as np 
from context_gym import ContextEnvironment


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

class AcrobotWrapper(ContextEnvironment):
    
    def __init__(self, env, history_len, clip_system_params, normalize_system_params, sampling_config=SAMPLING_UNIFORM):
        self.ALL_PARAMS  = {
                'LINK_MASS_1' : [0.1, 2.5],
                'LINK_MASS_2' : [0.1, 2.5],
                'LINK_LENGTH_1' : [0.3, 2.5],
                'LINK_LENGTH_2' : [0.3, 2.5],
            }
        self.system_params = list(sampling_config['params'].keys())
        self.sampling_config = sampling_config
        super().__init__(env, history_len, clip_system_params, normalize_system_params)
        assert len(set(self.system_params) - set(self.ALL_PARAMS.keys())) == 0

    
    def set_context(self, context):
        if self.normalize_system_params:
            # recover the context
            INTERVALS = self.sampling_config['params']
            context = {k : INTERVALS[k][0] + v *  (INTERVALS[k][1] - INTERVALS[k][0])    for k,v in context.items()}
            
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
            

        
            