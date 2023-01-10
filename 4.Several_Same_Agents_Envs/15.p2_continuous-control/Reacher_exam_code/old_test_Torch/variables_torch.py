import tensorflow as tf
import numpy      as np
import math

# -----------------------------------------------------

class Variables () :

    def __init__ ( self , state_size = 33 , action_size =  4 , hidden  =  512 ,

                          max_try = 2048 ,  cpus = 20 , elite = 20 ,

                          envs = None , brain_name = '' , beta = 0.01 , clip_size = 0.2 ,

                          batch_size = 32 , verbosity = 10 ) :

        self.action_size = action_size
        self.state_size  = state_size
        self.hidden      = hidden

        self.clip_size    = clip_size
        self.max_try      = max_try
        self.beta         = beta

        self.cpus       = cpus
        self.elite      = elite
        self.envs       = envs
        self.brain_name = brain_name

        self.verbosity      = verbosity
        self.batch_size     = batch_size
        self.steps_to_train = max_try
        self.memory_deque   = self.steps_to_train * elite

        print ( 'action_size:' , self.action_size )
        print ( 'state_size:'  , self.state_size  )
        print ( 'hidden:'      , self.hidden      )

        print ( 'clip_size:'      , self.clip_size    )
        print ( 'beta/entropyW:'  , self.beta         )

        print ( 'cpus:'        , self.cpus        )
        print ( 'elite:'       , self.elite       )
        print ( 'brain_name:'  , self.brain_name  )

        print ( 'verbosity:'      , self.verbosity      )
        print ( 'steps_to_train:' , self.steps_to_train )
        print ( 'memory_deque:'   , self.memory_deque   )
        print ( 'batch_size:'     , self.batch_size     )

        print ( 'batch_length:' , self.memory_deque // self.batch_size )
