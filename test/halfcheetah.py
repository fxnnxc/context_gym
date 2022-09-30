import gym 
from context_gym import HalfCheetahWrapper
import numpy as np 

env = HalfCheetahWrapper(gym.make("HalfCheetah-v3"), ['gravity_z', 'body_mass_1'], 3)
for context in range(10):
    returns = [] 
    for i in range(50):
        done = False         
        returns.append(0)
        env.reset()
        count= 0 
        while not done:
            action = env.action_space.sample()
            action = np.array([0,0,0,0,0,0])
            env.tau = None
            ns, r, done, info = env.step(action)
            # env.render()
            count +=1 
            returns[-1] += r
            # print(env.init_values)
    print(env.get_context())
    print(np.mean(returns), np.std(returns))
    context = env.sample_context()
    env.set_context(context) 




        

    