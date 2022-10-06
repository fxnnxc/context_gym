
import gym 
import numpy as np 
import Box2D

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

class LunarLanderWrapper(gym.Wrapper):
    
    # defines the valid boundary of the system parameters 
    ALL_PARAMS  = {
        'gravity_y' : [-12.0, 0.0],  # toward downside 
        'gravity_x' : [-2.0, 2.0],   # toward left and right
    }
    
    def __init__(self, env, system_params, history_len, sampling_config=SAMPLING_UNIFORM, clip_system_params=False):
        super().__init__(env)
        self.size = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)
        self.observation_space = gym.spaces.Dict(
            {"state" : env.observation_space,
             "history_obs" : env.observation_space.__class__(-np.inf, np.inf, (history_len, *self.size)),
             "history_act" : env.action_space.__class__(-np.inf, np.inf, (history_len, *self.action_space.shape)),
             "context" : gym.spaces.Box(-np.inf, np.inf, (len(system_params), ))
             }
        )
        self.history_len = history_len
        self.prev_state = None 
        self.history_obs = np.zeros((history_len, *self.size))
        self.history_act = np.zeros((history_len, *self.action_space.shape))
        self.system_params = system_params 
        assert len(set(self.system_params) - set(LunarLanderWrapper.ALL_PARAMS.keys())) == 0
        self.sampling_config = sampling_config
        self.clip_system_params = clip_system_params

        
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        self.history_obs = np.concatenate([self.history_obs[1:],  self.prev_state.reshape(1,-1)], axis=0)
        self.history_act = np.concatenate([self.history_act[1:],  action.reshape(1,-1)], axis=0)
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
        self.history_act = np.zeros((self.history_len, *self.action_space.shape))
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
            INTERVALS = LunarLanderWrapper.ALL_PARAMS
            context = {k : np.clip(method(v), INTERVALS[k][0], INTERVALS[k][1])    for k,v in params.items()} 
        return context
    
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
            

        
            