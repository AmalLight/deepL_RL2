import tensorflow as tf
import numpy      as np
import threading

def experiences_state_action ( states , actions ) :

    # print ( 'states shape:'  , states. shape ) # 128 , 8
    # print ( 'actions shape:' , actions.shape ) # 128 , 1

    da_ritorno = np.concatenate ( [ states , actions ] , axis = 1 ) # != np.append (..) ?

    # print ( 'da_ritorno shape:' , da_ritorno.shape ) # 128 , 9

    return da_ritorno # new states

from collections import deque
import random

class ReplayBuffer () :

    def __init__     ( self , settings ) :
        self.settings       = settings

        self.memory = deque ( maxlen = settings.BUFFER_SIZE )
        self.batch_size =              settings.BATCH_SIZE
    
    def add                ( self , state , action , reward , next_state , done   ) :
        self.memory.append (      ( state , action , reward , next_state , done ) )
    
    def sample ( self , k , chosen_list = [] ) :

        experiences = None
        if chosen_list == [] : experiences = random.sample ( self.memory , k )
        else                 : experiences = chosen_list

        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in experiences :

            states      += [ exp [ 0 ] ]
            actions     += [ exp [ 1 ] ]
            rewards     += [ exp [ 2 ] ]
            next_states += [ exp [ 3 ] ]
            dones       += [ exp [ 4 ] ]

        return ( np.vstack (      states ) , np.vstack ( actions ) , np.vstack ( rewards ) ,
                 np.vstack ( next_states ) , np.vstack ( dones   ) )

    # ---------------------------------------------------------------------------

    def big_sample ( self , agent ) :

        if self.len_memory () < self.settings.BATCH_SIZE : return self.sample ( k = self.settings.BATCH_SIZE , chosen_list = [] )

        threads = [ None ] * self.settings.cpus
        losses  = [ 0    ] * self.settings.cpus
        samples = [ None ] * self.settings.cpus

        def thread_big_sample ( local_critic , target_critic , target_actor , cpu ) :

            len_sample = self.settings.BATCH_SIZE
            indexes    = random.sample ( self.range_memory () , len_sample )

            sample          = [ self.memory [ i ] for i in indexes ]
            samples [ cpu ] =        sample

            # ( state , action , reward , next_state , done )
            # ---------------------------------------------------------------------------

            states      = np.array ( [ sample [ i ][ 0 ] for i in range ( len_sample ) ] )
            actions     = np.array ( [ sample [ i ][ 1 ] for i in range ( len_sample ) ] )
            next_states = np.array ( [ sample [ i ][ 3 ] for i in range ( len_sample ) ] )

            local_rewards = local_critic.predict_series ( states = [ states , actions ] , verbose = 0 )

            target_actions = target_actor. predict_series ( states =   next_states                    , verbose = 0 )
            target_rewards = target_critic.predict_series ( states = [ next_states , target_actions ] , verbose = 0 )

            # 2 = reward , 4 = done . update for target_rewards

            target_rewards = [     sample [ i ][ 2 ] + ( self.settings.GAMMA * target_rewards [ i ] * \
                             ( 1 - sample [ i ][ 4 ] ) )                                              \
                                      for   i in range ( len_sample ) ]

            # ---------------------------------------------------------------------------

            states_loss = [ ( ( local_rewards [ i ] - target_rewards [ i ] ) ** 2 \
                                \
                                + self.settings.small_constant ) ** self.settings.alpha \
                                \
                                for i in range ( len_sample ) ]

            losses [ cpu ] = np.sum ( states_loss , axis = 0 ) [ 0 ]

        # ---------------------------------------------------------------------------

        for cpu in range ( self.settings.cpus ) :

            threads [ cpu ] = threading.Thread ( target = thread_big_sample ,
                                                   args = ( agent.critic_local , agent.critic_target , agent.actor_target , cpu , ) )
            threads [ cpu ].start ()

        for cpu in range ( self.settings.cpus ) : threads [ cpu ].join ()

        # ---------------------------------------------------------------------------

        sum_losses = np.sum ( np.array ( losses ) , axis = 0 )

        probs = np.array ( [ losses [ i ] / sum_losses for i in range ( self.settings.cpus ) ] )

        index = np.random.choice ( self.settings.cpus , 1 , p = probs ) [ 0 ]

        return self.sample ( k = 0 , chosen_list = samples [ index ] )

    def len_memory   ( self ) : return         len ( self.memory )
    def range_memory ( self ) : return range ( len ( self.memory ) )
