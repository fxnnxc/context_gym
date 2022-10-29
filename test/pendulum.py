import gym 
from context_gym import PendulumWrapper
import numpy as np 

env = PendulumWrapper(gym.make("Pendulum-v1"), 3, True, True)
for context in range(10):
    returns = [] 
    for i in range(5):
        done = False         
        returns.append(0)
        env.reset()
        count= 0 
        while not done:
            action = env.action_space.sample()
            # action = np.array([0,0,0,0,0,0])
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




        

    