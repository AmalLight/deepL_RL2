import tensorflow as tf
import numpy      as np
import threading

from collections import deque
import random

class ReplayBuffer () :

    def __init__     ( self , settings ) :
        self.settings       = settings
        self.maxlen         = settings.memory_deque

        self.memory   = deque ( maxlen = self.maxlen )
        self.memory_i = 0

    def add                ( self , state , action , reward , old_probs , advantages , values ) :
        self.memory.append (      ( state , action , reward , old_probs , advantages , values ) )

    def len_memory     ( self ) : return     len ( self.memory )
    def reset_memory_i ( self ) :                  self.memory_i = 0
    def destroy        ( self ) :                  self.memory = deque ( maxlen = self.maxlen )
    def shuffle_simple ( self ) : random.shuffle ( self.memory )
    def shuffle        ( self ) :

        for_each_batch_lenght = ( self.len_memory () // self.settings.batch_size )

        memory_tmp = deque ( maxlen = self.maxlen )

        range_step_cpus = list ( range ( 0 , self.len_memory () , for_each_batch_lenght ) )

        for i in range ( len ( range_step_cpus ) ) :

            id_start = random.sample ( range_step_cpus , 1 ) [ 0 ]
            range_step_cpus.remove ( id_start )

            for add_i in range ( id_start , id_start + for_each_batch_lenght ) :
                memory_tmp.append ( self.memory [ add_i ] )

        self.memory = None
        self.memory = memory_tmp

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def sample ( self ) :

        batch_return = []
        for batch in range ( self.len_memory () // self.settings.batch_size ) :

            batch_return  += [ self.memory [ self.memory_i ] ]
            self.memory_i += 1

        # random.shuffle ( batch_return )

        states , actions , rewards , probs , advantages , _ = zip ( * batch_return )

        return ( tf.convert_to_tensor ( states     , dtype = float ) , tf.convert_to_tensor ( actions , dtype = float ) ,
                 tf.convert_to_tensor ( rewards    , dtype = float ) , tf.convert_to_tensor ( probs   , dtype = float ) ,
                 tf.convert_to_tensor ( advantages , dtype = float ) )

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def big_sample ( self , agent ) :

        threads = [ None ] * self.settings.thread_cpus
        losses  = [ 0    ] * self.settings.thread_cpus
        samples = [ None ] * self.settings.thread_cpus

        def thread_big_sample ( Critic , cpu ) :

            samples [ cpu ] = self.sample ()

            # ( state , action , reward , next_state , done )
            # ---------------------------------------------------------------------------

            states , _ , rewards , _ , _ , _ = zip ( * samples [ cpu ] ) # values as predictions
            states  = np.array ( states  )
            rewards = np.array ( rewards )

            predictions = agent.Critic.model ( states ).numpy ()

            # ---------------------------------------------------------------------------

            states_loss = [ ( ( rewards [ i ] - predictions [ i ] ) ** 2 \
                                \
                                + self.settings.small_constant ) ** self.settings.alpha \
                                \
                                for i in range ( self.settings.batch_size ) ]

            losses [ cpu ] = np.sum ( states_loss , axis = 0 ) [ 0 ]

        # ---------------------------------------------------------------------------

        for cpu in range ( self.settings.thread_cpus ) :

            threads [ cpu ] = threading.Thread ( target = thread_big_sample , args = ( agent.Critic , cpu , ) )
            threads [ cpu ].start ()

        for cpu in range ( self.settings.thread_cpus ) : threads [ cpu ].join ()

        # ---------------------------------------------------------------------------

        sum_losses = np.sum ( np.array ( losses ) , axis = 0 )

        probs = np.array ( [ losses [ i ] / sum_losses for i in range ( self.settings.thread_cpus ) ] )

        index = np.random.choice ( self.settings.thread_cpus , 1 , p = probs ) [ 0 ]

        states , actions , rewards , probs , advantages , _ = zip ( * samples [ index ] )

        return ( tf.convert_to_tensor ( states     , dtype = float ) , tf.convert_to_tensor ( actions , dtype = float ) ,
                 tf.convert_to_tensor ( rewards    , dtype = float ) , tf.convert_to_tensor ( probs   , dtype = float ) ,
                 tf.convert_to_tensor ( advantages , dtype = float ) )
