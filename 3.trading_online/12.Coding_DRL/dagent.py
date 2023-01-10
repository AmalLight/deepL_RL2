import tensorflow as tf
import numpy      as np

from dmodel_actor  import QNetwork_Actor
from dmodel_critic import QNetwork_Critic

from replay import ReplayBuffer , experiences_state_action
from noise  import OUNoise

class Agent () :

    def __init__ ( self , settings ) :
        self.settings   = settings

        self.actor_target  = QNetwork_Actor  ( self.settings )
        self.actor_local   = QNetwork_Actor  ( self.settings )
        self.critic_target = QNetwork_Critic ( self.settings )
        self.critic_local  = QNetwork_Critic ( self.settings )
        self.memory        = ReplayBuffer    ( self.settings )

        self.noise = OUNoise ( settings.action_size , 1234 )

    # ---------------------------------------------------------------------------

    def act                               ( self , state                       ) :
        action = self.actor_local.predict (        state = state , verbose = 0 )

        action += self.noise.sample ()
        action  = ( action + 1.0 ) / 2.0

        return np.clip ( action , 0 , 1 ) # sigmoid, selling or not selling, maybe buy if I don't have nothing

    # ---------------------------------------------------------------------------

    def step            ( self , state , action , reward , next_state , done ) :
        self.memory.add (        state , action , reward , next_state , done )

        if self.memory.len_memory () > self.settings.BATCH_SIZE :
           self.learn ( self.memory.big_sample ( self ) )

    # ---------------------------------------------------------------------------

    # experience from memory
    def learn ( self , experiences ) :

        def TAU_update ( local , target ) :

            target_weights = [ ( 1.0 - self.settings.TAU ) * el for el in target.model.trainable_weights ]
            local_weights  = [         self.settings.TAU   * el for el in local. model.trainable_weights ]

            weights = [ el1 + el2 for el1 , el2 in zip ( target_weights , local_weights ) ]
            target.setWeights                          (                        weights )

        states , actions , rewards , next_states , dones = experiences

        # ---------------------------- update critic ---------------------------- #

        actions_target = self.actor_target. predict_series ( states =   next_states                    , verbose = 0 ) # quantities
        Q_targets      = self.critic_target.predict_series ( states = [ next_states , actions_target ] , verbose = 0 ) # rewards

        predictions = [ None ] * len ( states )

        for i , critic_single_value in enumerate ( Q_targets ) :

            value = rewards [ i ] + ( self.settings.GAMMA * critic_single_value * ( 1 - dones [ i ] ) )
            predictions     [ i ] =                                       value

        predictions = np.vstack ( predictions )

        self.critic_local.training ( states = [ states , actions ] , labels = predictions , verbose = 0 )

        # ---------------------------- update actor ---------------------------- #

        states_constant = tf.constant ( tf.convert_to_tensor ( states , dtype = float ) )

        with tf.GradientTape () as tape :
             tape.reset      ()

             actions_local = self.actor_local. model (   states_constant                   )
             predictions   = self.critic_local.model ( [ states_constant , actions_local ] )

             loss = - tf.math.reduce_mean ( predictions ) # money_RIGHT_last_side_for_difference

        self.actor_local.optimizer.minimize ( loss , var_list = self.actor_local.model.trainable_variables , tape = tape )
        # print ( 'loss of the local Actor:'  , loss ) # -money_RIGHT_last_side_for_difference is to be learned.

        # ----------------------- update target networks ----------------------- #

        TAU_update ( self.critic_local , self.critic_target )
        TAU_update (  self.actor_local ,  self.actor_target )
