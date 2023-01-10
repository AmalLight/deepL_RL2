import utils

financial_params, ac_params = utils.get_env_param ()

print ( 'financial_params:' , financial_params )
print ( 'ac_params:'        , ac_params        )

# ------------------------------------------------------------

import numpy as np

import syntheticChrissAlmgren as sca

from ddpg_agent  import Agent
from collections import deque

env = sca.MarketEnvironment ()

print ( 'state_size:'  , env.observation_space_dimension () )
print ( 'action_size:' , env.action_space_dimension      () )

agent = Agent ( state_size = env.observation_space_dimension () , action_size = env.action_space_dimension () , random_seed = 0 )

# ------------------------------------------------------------

lqt      = 60   # Set the liquidation time
n_trades = 60   # Set the number of trades
tr       = 1e-6 # Set trader's risk aversion

episodes = 10000 # Set the number of episodes to run the simulation

shortfall_hist  = np.array ( [] )
shortfall_deque = deque ( maxlen = 100 )

BUFFER_SIZE = int ( 1e4 ) # replay buffer size
BATCH_SIZE = 128          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR_ACTOR = 1e-4           # learning rate of the actor 
LR_CRITIC = 1e-3          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay

print ( 'liquidation time:'   , lqt         )
print ( 'number of trades:'   , n_trades    )
print ( 'risk aversion:'      , tr          )
print ( 'episodes:'           , episodes    )
print ( 'replay buffer size:' , BUFFER_SIZE )
print ( 'learn BATCH_SIZE:'   , BATCH_SIZE  )
print ( 'GAMMA discount:'     , GAMMA       )
print ( 'TAU update:'         , TAU         )

print ( 'Adam Learning rate for ACTOR:'  , LR_ACTOR     )
print ( 'Adam Learning rate for CRITIC:' , LR_CRITIC    )
print ( 'CRITIC WEIGHT_DECAY:'           , WEIGHT_DECAY )

# ------------------------------------------------------------

for episode in range ( episodes ) :

    cur_state = env.reset ( seed = episode , liquid_time = lqt , num_trades = n_trades , lamb = tr )

    env.start_transactions () # set the environment to make transactions, like press ENTER to start the game

    for i in range ( n_trades + 1 ) :

        action = agent.act ( cur_state , add_noise = True )

        new_state, reward, done, info = env.step ( action )

        agent.step ( cur_state , action , reward , new_state , done )

        cur_state = new_state

        if info.done :
            shortfall_hist = np.append ( shortfall_hist , info.implementation_shortfall )
            shortfall_deque.    append (                  info.implementation_shortfall )
            break
        
    print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format ( episode + 1 , episodes , np.mean ( shortfall_deque ) ) )
print ( '\nAverage Implementation Shortfall: ${:,.2f} \n'. format (                          np.mean ( shortfall_hist  ) ) )
