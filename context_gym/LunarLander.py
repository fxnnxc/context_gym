
import gym 
import numpy as np 
import Box2D
from context_gym import ContextEnvironment



SAMPLING_NORMAL = {
    "sample" : lambda v : np.random.normal(v[0], v[1]),
    "params":{
        'gravity_y' : [-9.8, 1.0],  # toward downside 
        'gravity_x' : [0.0, 1.0]
    }
}
SAMPLING_UNIFORM = {
    "sample" : lambda v : np.random.uniform(v[0], v[1]),
    "params":{
        'gravity_y' : [-10.0, -8.0],  # toward downside 
        'gravity_x' : [-1.0, 1.0]
    }
}

class LunarLanderContinuousWrapper(ContextEnvironment):
    
    # defines the valid boundary of the system parameters 
    
    
    def __init__(self, env, history_len, clip_system_params, normalize_system_params, sampling_config=SAMPLING_UNIFORM):
        self.ALL_PARAMS  = {
                'gravity_y' : [-12.0, 0.0],  # toward downside 
                'gravity_x' : [-2.0, 2.0],   # toward left and right
            }
            
        self.system_params = list(sampling_config['params'].keys())
        self.sampling_config = sampling_config
        super().__init__(env, history_len, clip_system_params, normalize_system_params)
        assert len(set(self.system_params) - set(self.ALL_PARAMS.keys())) == 0
    
    def set_context(self, context):
        # define how to set environment variables 
        if "gravity_y" in self.system_params:
            origin = self.env.unwrapped.world.gravity
            self.env.unwrapped.world.gravity = (origin[0], context['gravity_y']) # no X gravity
        if "gravity_x" in self.system_params:
            origin = self.env.unwrapped.world.gravity
            self.env.unwrapped.world.gravity = (context['gravity_x'], origin[1]) # no X gravity

    def get_context(self):
        context = {} 
        if "gravity_x" in self.system_params:
            context['gravity_x'] = self.env.unwrapped.world.gravity[0]     
        if "gravity_y" in self.system_params:
            context['gravity_y'] = self.env.unwrapped.world.gravity[1]     

        return context 
    
    
if __name__ == "__main__":
    env = LunarLanderWrapper(gym.make("LunarLanderContinuous-v2"), ['gravity_x', 'gravity_y'], 3, sampling_config=SAMPLING_NORMAL)
    
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
        print(count)
            

        
            