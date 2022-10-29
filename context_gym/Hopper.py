
import gym 
import numpy as np 
import Box2D
from context_gym import ContextEnvironment


SAMPLING_NORMAL = {
    "sample" : lambda v : np.random.normal(v[0], v[1]),
    "params":{
        'gravity_z' : [-9.8, 1.0],  # toward downside 
        'gravity_x' : [0.0, 1.0],
        'gravity_y' : [0.0, 1.0],
        'body_mass_1' : [3.53-1.0, 3.53+1.0], 
        'body_mass_2' : [3.92-1.0, 3.92+1.0], 
        'body_mass_3' : [2.71-0.8, 2.71+0.8], 
        'body_mass_4' : [5.08+2.0, 5.08-2.0],  
    }
}
SAMPLING_UNIFORM = {
    "sample" : lambda v : np.random.uniform(v[0], v[1]),
    "params":{
        'gravity_z' : [-10.0, -8.0],  # toward downside 
        'gravity_x' : [-1.0, 1.0],
        'gravity_y' : [-1.0, 1.0],
        'body_mass_1' : [3.53-1.0, 3.53+1.0], 
        'body_mass_2' : [3.92-1.0, 3.92+1.0], 
        'body_mass_3' : [2.71-0.8, 2.71+0.8], 
        'body_mass_4' : [5.08+2.0, 5.08-2.0], 
    }
}

class HopperWrapper(ContextEnvironment):
 
    
    def __init__(self, env, history_len, clip_system_params, normalize_system_params, sampling_config=SAMPLING_UNIFORM):
        self.ALL_PARAMS  = {
            'gravity_z' : [-12.0, 0.0],  # toward downside 
            'gravity_x' : [-2.0, 2.0],   # invalid beacuse it moves the robot to direction 
            'gravity_y' : [-0.0, 0.0],   # invalid beacuse it moves the robot to direction   
            'body_mass_1' : [3.53-1.0, 3.53+1.0], 
            'body_mass_2' : [3.92-1.0, 3.92+1.0], 
            'body_mass_3' : [2.71-0.8, 2.71+0.8], 
            'body_mass_4' : [5.08+2.0, 5.08-2.0], 
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
        if "gravity_x" in self.system_params:
            origin = self.env.model.opt.gravity
            self.env.model.opt.gravity[0] = context['gravity_x']  # no X gravity
        if "gravity_y" in self.system_params:
            origin = self.env.model.opt.gravity
            self.env.model.opt.gravity[1] = context['gravity_y']  # no Y gravity
        if "gravity_z" in self.system_params:
            origin = self.env.model.opt.gravity
            self.env.model.opt.gravity[2] = context['gravity_z']  # no Y gravity
        for i in range(1, 5):
            if f'body_mass_{i}' in self.system_params:
                origin = self.env.model.body_mass
                self.env.model.body_mass[i] = context[f'body_mass_{i}']  # no Y gravity
        return 

    def get_context(self):
        context = {} 
        if "gravity_x" in self.system_params:
            context['gravity_x'] = self.env.model.opt.gravity[0]  
        if "gravity_y" in self.system_params:
            context['gravity_y'] = self.env.model.opt.gravity[1]
        if 'gravity_z' in self.system_params:
            context['gravity_z'] = self.env.model.opt.gravity[2]     
        if 'body_mass_1' in self.system_params:
            context['body_mass_1'] = self.env.model.body_mass[1]
        for i in range(1, 5):
            if f'body_mass_{i}' in self.system_params:
                context[f'body_mass_{i}'] = self.env.model.body_mass[i]
                
        return context 
    
    
if __name__ == "__main__":
    env = HopperWrapper(gym.make("Hopper-v3"), ['gravity_x', 'gravity_y'], 3, sampling_config=SAMPLING_NORMAL)
    
    for i in range(100):
        done = False         
        context = env.sample_context()
        env.set_context(context)
        print(env.get_context())
        
        env.reset()
        count= 0 
        while not done:
            action = env.action_space.sample()
            env.tau = None
            ns, r, done, info = env.step(action)
            # env.render()
            count +=1 
            print(dir(env.model))
            print(env.model.body_mass)
            assert False 
        print(count)
            

        
            