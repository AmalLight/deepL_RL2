from unityagents import UnityEnvironment
import numpy as np

envs = UnityEnvironment ( file_name = '../Reacher_Linux/Reacher.x86_64' )

# get the default brain
brain_name = envs.brain_names [      0     ]
brain      = envs.brains      [ brain_name ]

# -----------------------------------------------------------------------------------

# reset the environment
envs_info = envs.reset ( train_mode = True ) [ brain_name ]

# number of agents
num_agents = len (      envs_info.agents )
print ( 'Number of agents:' , num_agents ) # 20

# size of each action
action_size = brain.vector_action_space_size
print ( 'Size of each action:' , action_size ) # 4

# examine the state space 
states     = envs_info.vector_observations
state_size = states.shape [ 1 ]

print ( 'There are {} agents. Each observes a state with length: {}'.format ( states.shape [ 0 ] , state_size ) ) # 20,33
print ( 'The state for the first agent looks like:'                         , states       [ 0 ]                ) # ...

# -----------------------------------------------------------------------------------

from variables_torch import Variables

settings = Variables ( state_size = state_size , action_size = action_size ,

                       cpus = num_agents , envs = envs , brain_name = brain_name )

from agent_torch import Agent
future =                Agent ( settings )

future.cross_entropy_loss ()
envs.close                ()
