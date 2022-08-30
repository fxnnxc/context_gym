
import gym 
import numpy as np 

class CartPoleWrapper(gym.Wrapper):
    
    ALL_PARAMS  = [
        'gravity',
        'length'
    ]
    
    def __init__(self, env, system_params, history_len):
        super().__init__(env)
        size = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)
        self.observation_space = gym.spaces.Dict(
            {"state" : env.observation_space,
             "history" : env.observation_space.__class__(-np.inf, np.inf, (history_len, *size)),
             "context" : gym.spaces.Box(-np.inf, np.inf, (2,)) 
             }
        )
        self.history = np.zeros((history_len, *size))
        self.system_params = system_params 
        assert len(set(self.system_params) - set(CartPoleWrapper.ALL_PARAMS)) == 0
        
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        self.history = np.concatenate([self.history[1:],  next_state.reshape(1,-1)], axis=0)
        next_state = {
            "state" : next_state,
            "history" : self.history.copy(),
            "context" : self.get_context()['gravity']
        }
        return next_state, reward, done, info
    def reset(self):
        state =  super().reset()
        self.history = np.concatenate([self.history[1:],  state.reshape(1,-1)], axis=0)
        state = {
            "state" : state,
            "history" : self.history.copy(),
            "context" : self.get_context()['gravity']
        }
        return state 
    
    def sample_context(self):
        # generate random context
        context = {
            "gravity" : np.random.random()*9.8,
            'length' : np.random.random() + 0.5 
        } 
        return context
    
    def set_context(self, context):
        # define how to set environment variables 
        if "gravity" in self.system_params:
            self.env.unwrapped.gravity = context['gravity']
        if 'length' in self.system_params:
            self.env.unwrapped.length = context['length']
        
    def get_context(self):
        context = {} 
        if "gravity" in self.system_params:
            context['gravity'] = self.env.unwrapped.gravity
        if 'length' in self.system_params:
            context['length'] = self.env.unwrapped.length
        return context 
    
    
if __name__ == "__main__":
    env = CartPoleWrapper(gym.make("CartPole-v1"), ['gravity', 'length'], 3)
    # env = gym.make("CartPole-v1")
    # print(dir(env.unwrapped))
    # env.length = 0
    
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
            

        
            