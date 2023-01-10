import tensorflow as tf
import numpy      as np
import math

# -----------------------------------------------------

class Variables () :

    def __init__ ( self , state_size = 33 , action_size = 4 , hidden = 512 , conv_kernel = 0 ,

                          max_try = 2048 , basic = tf.nn.tanh , final = tf.nn.tanh ,

                          cpus = 20 , elite = 20 , thread_cpus = 4 , vector = 1 ,

                          envs = None , alpha = 0.4 , brain_name = '' , beta = 0.01 , clip_size = 0.2 ,

                          small_constant = 0.01 , batch_size = 32 , verbosity = 1 , quality = 0.01 ) :

        self.action_size = action_size
        self.state_size  = state_size
        self.hidden      = hidden
        self.conv_kernel = conv_kernel

        self.max_try   = max_try
        self.clip_size = clip_size
        self.beta      = beta

        self.basic = basic
        self.final = final

        self.cpus         = cpus
        self.elite        = elite
        self.thread_cpus  = thread_cpus

        self.envs        = envs
        self.brain_name  = brain_name
        self.vector_size = vector

        self.quality        = quality
        self.alpha          = alpha
        self.verbosity      = verbosity
        self.small_constant = small_constant

        self.batch_size     = batch_size
        self.steps_to_train = max_try
        self.memory_deque   = self.steps_to_train * elite * vector

        print ( 'action_size:'   , self.action_size )
        print ( 'state_size:'    , self.state_size  )
        print ( 'hidden:'        , self.hidden      )
        print ( 'conv_kernel:'   , self.conv_kernel )
        print ( 'clip_size:'     , self.clip_size   )
        print ( 'beta/entropyW:' , self.beta        )

        print ( 'basic activation Dense:' , self.basic )
        print ( 'final activation Actor:' , self.final )

        print ( 'cpus:'        , self.cpus        )
        print ( 'elite:'       , self.elite       )
        print ( 'thread_cpus:' , self.thread_cpus )
        print ( 'brain_name:'  , self.brain_name  )

        print ( 'quality:'        , self.quality        )
        print ( 'alpha:'          , self.alpha          )
        print ( 'verbosity:'      , self.verbosity      )
        print ( 'small_constant:' , self.small_constant )

        print ( 'steps_to_train:' , self.steps_to_train )
        print ( 'vector_size:'    , self.vector_size    )
        print ( 'memory_deque:'   , self.memory_deque   )
        print ( 'batch_size:'     , self.batch_size     )

        print ( 'batch_length:' , self.memory_deque // self.batch_size )
