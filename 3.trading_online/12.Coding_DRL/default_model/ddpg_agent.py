import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE  = int ( 1e4 ) # replay buffer size
BATCH_SIZE   = 128         # minibatch size
GAMMA        = 0.99        # discount factor
TAU          = 1e-3        # for soft update of target parameters
LR_ACTOR     = 1e-4        # learning rate of the actor 
LR_CRITIC    = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0           # L2 weight decay

device = torch.device ( "cuda:0" if torch.cuda.is_available () else "cpu" )

class Agent () :
    
    def __init__ ( self , state_size , action_size , random_seed ) :
        """
        state_size  (int): dimension of each state
        action_size (int): dimension of each action
        random_seed (int): random seed
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.seed        = random.seed ( random_seed )

        self.actor_local  = Actor ( state_size , action_size , random_seed ).to ( device )
        self.actor_target = Actor ( state_size , action_size , random_seed ).to ( device )

        self.critic_local  = Critic ( state_size , action_size , random_seed ).to ( device )
        self.critic_target = Critic ( state_size , action_size , random_seed ).to ( device )

        self.actor_optimizer  = optim.Adam ( self.actor_local.parameters  () , lr = LR_ACTOR                                )
        self.critic_optimizer = optim.Adam ( self.critic_local.parameters () , lr = LR_CRITIC , weight_decay = WEIGHT_DECAY )

        # Noise process
        self.noise = OUNoise ( action_size , random_seed )

        # Replay memory
        self.memory = ReplayBuffer ( action_size , BUFFER_SIZE , BATCH_SIZE , random_seed )

    def step            ( self , state , action , reward , next_state , done ) :
        self.memory.add (        state , action , reward , next_state , done )

        if len ( self.memory ) > BATCH_SIZE :
           experiences = self.memory.sample ()
           self.learn ( experiences , GAMMA )

    def act ( self , state , add_noise = True ) :
        # play with Actor = quantities definitions

        state = torch.from_numpy ( state ).float ().to ( device )
        self.actor_local.eval () # training off

        with torch.no_grad () : action = self.actor_local ( state ).cpu ().data.numpy ()
        self.actor_local.train () # training on

        if add_noise : action += self.noise.sample ()
        action = ( action + 1.0 ) / 2.0

        return np.clip ( action , 0 , 1 )

    def reset ( self ) : self.noise.reset ()

    def learn ( self , experiences , gamma ) :
        """
        Q_targets = r + γ * critic_target ( next_state , actor_target ( next_state ) )
        actor_target  ( state          ) -> action
        critic_target ( state , action ) -> Q-value

        experiences = [ Tuple of (s, a, r, s', done) ]
        gamma       =   float of discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #

        actions_next   = self.actor_target  ( next_states                ) # actions_next = list quantities definitions <- [ q1 , q2 , .. ] | qx for stock_x
        Q_targets_next = self.critic_target ( next_states , actions_next )

        # Q_targets_next = states + quantities = list rewards value <- [ r1 , r2 , .. ] | rx for stock_x

        # Q_targets_next is a reward(s), in this case len ( Q_targets_next ) == 1 because we have only 1                                stock_id
        # Q_targets_next get actions_next as params/variables for each stock_x, in this case take only 1 quantity as param/variable for stock_id

        # to have len ( Q_targets_next ) > 1 we need a states as matrix where rows are stocks_id

        Q_targets = rewards + ( gamma * Q_targets_next * ( 1 - dones ) ) # from DDQN

        Q_expected  = self.critic_local ( states , actions )
        critic_loss = F.mse_loss ( Q_expected , Q_targets )

        self.critic_optimizer.zero_grad ()
        critic_loss.backward            ()
        self.critic_optimizer.step      ()

        # ---------------------------- update actor ---------------------------- #

        actions_pred =   self.actor_local  ( states                )
        actor_loss   = - self.critic_local ( states , actions_pred ).mean ()

        self.actor_optimizer.zero_grad ()
        actor_loss.backward            ()
        self.actor_optimizer.step      ()

        # ----------------------- update target networks ----------------------- #

        self.soft_update ( self.critic_local , self.critic_target , TAU )
        self.soft_update (  self.actor_local , self.actor_target  , TAU )                     

    def soft_update ( self , local_model , target_model , tau ) :

        for target_param , local_param in zip ( target_model.parameters () , local_model.parameters () ) :
            target_param.data.copy_ ( tau * local_param.data + ( 1.0 - tau ) * target_param.data )

class OUNoise:

    def __init__ ( self , size , seed , mu = 0.0 , theta = 0.15 , sigma = 0.2 ) :

        self.mu = mu * np.ones ( size )
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed ( seed )
        self.reset ()

    def reset  ( self ) : self.state = copy.copy ( self.mu )
    def sample ( self ):

        x  = self.state
        dx = self.theta * ( self.mu - x ) + self.sigma * np.array ( [ random.random () for i in range ( len ( x ) ) ] )
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__ ( self ) : return len ( self.memory )
