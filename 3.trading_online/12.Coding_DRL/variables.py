class Variables () :

    def __init__ ( self ) :

        self.BUFFER_SIZE = int ( 1e4 ) # replay buffer size
        self.BATCH_SIZE  = 128         # minibatch size
        self.GAMMA       = 0.99        # discount factor
        self.TAU         = 1e-3        # for soft update of target parameters
        self.LR_ACTOR    = 1e-4        # learning rate of the actor 
        self.LR_CRITIC   = 1e-3        # learning rate of the critic

        print ( 'replay buffer size:' , self.BUFFER_SIZE )
        print ( 'learn BATCH_SIZE:'   , self.BATCH_SIZE  )
        print ( 'GAMMA discount:'     , self.GAMMA       )
        print ( 'TAU update:'         , self.TAU         )

        print ( 'Adam Learning rate for ACTOR:'  , self.LR_ACTOR  )
        print ( 'Adam Learning rate for CRITIC:' , self.LR_CRITIC )

        self.small_constant = 0.01 # it enables to make division by 0
        self.alpha = 0.4           # usually exponential function increases a value exponentially for each x ** a
                                   # in this case more x is bigger more it will be increased slowly
                                   # a little value will be never like a bigger, but
                                   # more smaller values could reach ( sum ) a bigger ( one )
        self.cpus = 7

        print ( 'small_constant:' , self.small_constant )
        print ( 'alpha:'          , self.alpha          )
        print ( 'cpus:'           , self.cpus           )

        # WEIGHT_DECAY = 0 # L2 weight decay
        # print ( 'CRITIC WEIGHT_DECAY:' , WEIGHT_DECAY )

    def set_vars ( self  , state_size , action_size , units1 = 24 , units2 = 48 ) :

        self.state_size  = state_size
        self.action_size = action_size

        self.units1 = units1
        self.units2 = units2

        # print ( 'state_size:'  , self.state_size  )
        # print ( 'action_size:' , self.action_size )

        print ( 'units1:' , self.units1 )
        print ( 'units2:' , self.units2 )
