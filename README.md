# context_gym


Simple Domain Randomization using wrapper class


```bash
pip insatll -e .
```


## Update 

* 🌟 2022.10.06 : added `clip_system_params` (default is `False`) when sampling context



## Sampling contex and Setting the context

You can set context by (1) sampling and (2) setting

```python
context = env.sample_context()
env.set_context(context) 
```


## Python Example

Here is full code for HalfCheetah Environment. 
For other environments, you can copy and implement the environment

```python
import gym 
from context_gym import HalfCheetahWrapper


SAMPLING_UNIFORM = {
    "sample" : lambda v : np.random.uniform(v[0], v[1]),  # how to sample in the interval
    "params":{
        'gravity_z' : [-10.0, -8.0],  # toward downside 
        'gravity_x' : [-1.0, 1.0],
        'gravity_y' : [-1.0, 1.0],
        'body_mass_1' : [6.36-1.0, 6.36+1.0 ]     
    }
}

env = HalfCheetahWrapper(
                    env=gym.make("HalfCheetah-v3"), 
                    system_params=['gravity_z', 'body_mass_1'], 
                    history_len=3, 
                    sampling_config=SAMPLING_UNIFORM
        )

"""
For each context, run 50 sample episodes.
"""
n_contex = 10
for context in range(n_contex):
    # randomize contextes
    context = env.sample_context()
    env.set_context(context) 

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
            count +=1 
            returns[-1] += r
    print(env.get_context())
    print(np.mean(returns), np.std(returns))
    # how to set context

```
