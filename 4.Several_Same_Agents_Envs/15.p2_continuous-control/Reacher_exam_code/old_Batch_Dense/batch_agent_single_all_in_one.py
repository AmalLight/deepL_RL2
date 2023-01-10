import tensorflow             as tf
import tensorflow_probability as tfp
import numpy                  as np

# pip3           install           tensorflow_probability
# python3 -m pip install --upgrade tensorflow

import random , sys , threading , math , time , gym

from unityagents import UnityEnvironment
from collections import deque

from model_single_all_in_one import Network_Input_Value_Action
from replay                  import ReplayBuffer

# ---------------------------------------------------------------------------
# A2C ( sync ) A3C ( async )
# Big Deep Network == dense matrix.
# dense matrix   for MC and TD is not possible ( in continuous )
# states,actions for MC and TD is not possible ( in continuous )
# is not possible to evaluate infinite actions without a minimal buffer

class Agent () :

    def __init__ ( self , settings ) :
        self.settings   = settings
        self.collection_states      = []
        self.collection_next_states = []
        self.collection_continuosV  = []
        self.collection_dones       = []
        self.collection_rewards     = []
        self.collection_values      = []
        self.collection_probs       = []

        self.collection_rewards_step = np.zeros ( self.settings.cpus )

    def play_EP ( self , ie = 0 , envs_info = None ) :

        self.collection_rewards_sum = np.zeros ( self.settings.cpus )

        states = np.array ( envs_info.vector_observations )

        # norm on axis +0 -> connection between all stack's rows    --> Rescale + Normalize_the_stack => 1,33
        # norm on axis -1 -> connection between all stack's vectors --> Rescale only                  => 20,1

        # states /= np.linalg.norm ( states , axis=0 , keepdims=True ) + self.settings.small_constant

        for t in range ( self.settings.max_try ) :

            # ------------------------------------------------------

            mean = self.network.model ( [ states ] ) [ 1 ].numpy () # ( 20 , 4 )

            tfp_dist   = tfp.distributions.Normal ( loc = mean , scale = tf.math.softplus ( self.std ) )
            continuosV = tfp_dist.sample    ().numpy ()
            old_probs  = tfp_dist.log_prob  ( continuosV           )
            old_probs  = tf.math.reduce_sum ( old_probs , axis = 1 )

            # ------------------------------------------------------

            envs_info = self.settings.envs.step ( continuosV ) [ self.settings.brain_name ]

            next_states = np.array ( envs_info.vector_observations )
            rewards     = np.array ( envs_info.rewards             )
            dones       = np.array ( envs_info.local_done          )

            # next_states /= np.linalg.norm ( next_states , axis=0 , keepdims=True ) + self.settings.small_constant 

            # ------------------------------------------------------
            # ------------------------------------------------------

            self.collection_continuosV. append ( continuosV                                   ) # append on tail
            self.collection_states.     append ( states                                       ) # append on tail
            self.collection_next_states.append ( next_states                                  ) # append on tail
            self.collection_dones.      append ( dones                                        ) # append on tail
            self.collection_values.     append ( self.network.model ( states ) [ 0 ].numpy () ) # append on tail
            self.collection_probs.      append ( old_probs                                    ) # append on tail

            self.collection_rewards_step += rewards
            self.collection_rewards_sum  += rewards

            # rewards /= np.linalg.norm ( rewards , axis=0 , keepdims=True ) + self.settings.small_constant

            self.collection_rewards.append ( rewards ) # append on tail

            states = next_states

            print ( '\rEP:' , ie + 1 , 't:' , t + 1 ,
                      'EP:'   , round ( np.mean ( self.collection_rewards_sum  ) , 3 ) ,
                      'Step:' , round ( np.mean ( self.collection_rewards_step ) , 3 ) ,
                      'MEM:'  ,   len (           self.collection_states       )       , end='' )

            # ------------------------------------------------------
            # ------------------------------------------------------

            if ( any ( dones ) or ( ( ( t + 1 ) %    self.settings.steps_to_train == 0  ) ) ) and \
               ( len ( self.collection_states ) == ( self.settings.memory_deque           ) ) :

               real_steps_to_train = np.array ( self.collection_rewards ).shape [ 0 ]

               self.states_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size  ) )
               self.next_states_elite = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size  ) )
               self.continuosV_elite  = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.action_size ) )
               self.rewards_elite     = np.zeros ( ( real_steps_to_train , self.settings.elite                             ) )
               self.dones_elite       = np.ones  ( ( real_steps_to_train , self.settings.elite                             ) )
               self.values_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , 1                         ) )
               self.probs_elite       = np.zeros ( ( real_steps_to_train , self.settings.elite                             ) )

               print ( '' ) ; self.training_for_elite (real_steps_to_train) ; print ( '' )

               self.collection_states      = []
               self.collection_next_states = []
               self.collection_continuosV  = []
               self.collection_dones       = []
               self.collection_rewards     = []
               self.collection_values      = []
               self.collection_probs       = []

               self.collection_rewards_step = np.zeros ( self.settings.cpus )
               self.replay.destroy ()

            if any ( dones ) : break

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def advantage_rewards ( self , rewards , real_steps_to_train ) :
        advantage_rewards = np.zeros     ( ( real_steps_to_train , self.settings.elite , 1 ) )

        # self.dones_elite = np.expand_dims ( self.dones_elite  , 2 )
        rewards            = np.expand_dims (           rewards , 2 )

        # --------------------------------------------------------------

        for i in reversed ( range ( real_steps_to_train ) ) :

            old_TD_state_action  = self.values_elite [ i     ]
            next_TD_state_action = self.values_elite [ i + 1 ] if ( i < ( real_steps_to_train - 1 ) ) \
                                                               else self.network.model ( self.next_states_elite [ -1 ] )[ 0 ].numpy ()

            error = rewards [ i ] + next_TD_state_action - old_TD_state_action # without gamma and dones

            advantage_rewards [ i ] += ( advantage_rewards [ i+1 ] if ( i < ( real_steps_to_train - 1 ) ) else 0 )

        # --------------------------------------------------------------

        rewards_mean = np.mean ( advantage_rewards , axis = 0 ) # self.settings.elite,1
        rewards_std  = np.std  ( advantage_rewards , axis = 0 ) + self.settings.small_constant

        advantage_rewards -= rewards_mean
        advantage_rewards /= rewards_std

        return advantage_rewards

    def future_rewards ( self , rewards ) :

        rewards [ -1 ] += np.squeeze ( self.network.model ( self.next_states_elite [ -1 ] )[ 0 ].numpy () , axis = -1 ) # structural recursion +-

        # for  row in range                   ( rewards.shape   [  0  ]            ) : rewards [ row , : ] *= ( 0.955 ** row )
        for    col in range                   ( rewards.shape   [  1  ]            ) :
           for row in range                   ( rewards.shape   [  0  ]            ) :
               rewards [ row , col ] = np.sum ( rewards [ row : , col ] , axis = 0 ) # classic future rewards

        # rewards_mean = np.mean ( rewards , axis = 1 , keepdims=True ) # real_steps_to_train,1
        # rewards_std  = np.std  ( rewards , axis = 1 , keepdims=True ) + self.settings.small_constant

        # rewards -= rewards_mean
        # rewards /= rewards_std
        return       rewards # no norm if added the next_states' value, because norm contains other rows that now are missing

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training_for_elite ( self , real_steps_to_train ) :

        elite_idxs = self.collection_rewards_step.argsort () [ - self.settings.elite : ] # it Returns max values => direct progress

        for i , e in enumerate ( elite_idxs ) :

            self.states_elite      [ : , i ] = np.array ( self.collection_states      ) [ : , e ]
            self.next_states_elite [ : , i ] = np.array ( self.collection_next_states ) [ : , e ]
            self.continuosV_elite  [ : , i ] = np.array ( self.collection_continuosV  ) [ : , e ]
            self.rewards_elite     [ : , i ] = np.array ( self.collection_rewards     ) [ : , e ]
            # self.dones_elite     [ : , i ] = np.array ( self.collection_dones       ) [ : , e ] - 1.0
            self.values_elite      [ : , i ] = np.array ( self.collection_values      ) [ : , e ]
            self.probs_elite       [ : , i ] = np.array ( self.collection_probs       ) [ : , e ]

        # --------------------------------------------------------------

        self.future_rewards_elite    = self.future_rewards    ( self.rewards_elite                       )
        self.advantage_rewards_elite = self.advantage_rewards ( self.rewards_elite , real_steps_to_train )

        self.future_rewards_elite    = np.expand_dims ( self.future_rewards_elite    , -1                  )
        self.advantage_rewards_elite = np.expand_dims ( self.advantage_rewards_elite , -1                  )
        self.training                                 ( self.future_rewards_elite    , real_steps_to_train )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training ( self , rewards , real_steps_to_train ) :

        # https://github.com/jknthn/reacher-ppo/blob/master/agent.py
        # https://github.com/chris838/reacher/blob/master/ppo_agent.py
        # https://github.com/bonniesjli/PPO-Reacher_UnityML/blob/master/agent.py

        # from actions to probs, but not dense as MC ( i am using Gaussian distribution )
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal
        # distributions Normal is == np.random.normal

        # ------------------------------------------------------------------------------------------

        for i in range      ( self.states_elite.shape      [ 0 ] ) :
            self.replay.add ( self.states_elite            [ i ] ,
                              self.continuosV_elite        [ i ] ,
                                   rewards                 [ i ] ,
                              self.probs_elite             [ i ] ,
                              self.advantage_rewards_elite [ i ] ,
                              self.values_elite            [ i ] ) # append on tail

        total = self.settings.verbosity * ( self.replay.len_memory () // self.settings.batch_size )

        self.network.lr_schedule.addNewSteps ( real_steps_to_train )

        # pure lr => go well at first, go bad after few.
        # high lr learns all ( included little advs ), a lower lr learns only by the best advs
        # set more batches is a cheat to use 512 size on old computers

        print ( 'memory:' , self.replay.len_memory () )
        iter_step = 0 ; print ( '\r' , iter_step , '/' , total , end='' , sep='' )

        if self.replay.len_memory () < self.settings.batch_size : return None

        # ------------------------------------------------------------------------------------------

        std = tf.Variable ( self.std , dtype = float , name = 'std' )

        for v in range ( self.settings.verbosity ) :

            self.replay.reset_memory_i ()
            self.replay.shuffle        ()

            for i in range ( self.replay.len_memory () // self.settings.batch_size ) :
                iter_step += 1

                sample = self.replay.sample ()

                stats  = []
                acts   = []
                futurs = []
                probs  = []
                advs   = []

                for els in sample :
                    stats  += [ els [ 0 ] ]
                    acts   += [ els [ 1 ] ]
                    futurs += [ els [ 2 ] ]
                    probs  += [ els [ 3 ] ]
                    advs   += [ els [ 4 ] ]

                stats  = tf.convert_to_tensor ( stats  , dtype = float )
                acts   = tf.convert_to_tensor ( acts   , dtype = float )
                futurs = tf.convert_to_tensor ( futurs , dtype = float )
                probs  = tf.convert_to_tensor ( probs  , dtype = float )
                advs   = tf.convert_to_tensor ( advs   , dtype = float )

                # https://www.tensorflow.org/api_docs/python/tf/clip_by_norm # it is == np.clip with only max value + Norm

                with tf.GradientTape ( persistent = False ) as tape :
                     tape.reset ()

                     value_predictions , mean_local = self.network.model ( stats ) # MSE , 0.5 => 50% trust

                     # --------------------------------------------------------------------------------------------------

                     tfp_dist  = tfp.distributions.Normal ( loc = mean_local , scale = tf.math.softplus ( std ) )

                     new_probs = tfp_dist.log_prob  ( acts                            ) # <= 0
                     new_probs = tf.math.reduce_sum ( new_probs           , axis = -1 ) # sum log probs == log ( mult ( probs ) ) == log for AND
                     entropy   = tf.math.reduce_sum ( tfp_dist.entropy () , axis = -1 ) # Shannon entropy in nats = entropy using e

                     old_probs = tf.math.exp (     probs            )
                     new_probs = tf.math.exp ( new_probs            ) # log_prob must be log based on 2 or e
                     ratio     =               new_probs / old_probs  # >= 0

                     ratio = tf.math.reduce_min   (
                             tf.convert_to_tensor ( [ ratio                                                         * advs , \
                             tf.clip_by_value     (   ratio , clip_value_min = 1 - 0.2 , clip_value_max = 1 + 0.2 ) * advs ] ) , axis = 0 )

                     loss = (  tf.math.reduce_mean ( tf.math.pow ( futurs - value_predictions , 2 ) ) * 0.5        ) + \
                            ( -tf.math.reduce_mean ( ratio ) -self.settings.beta * tf.math.reduce_mean ( entropy ) )

                # watch original -R as +Loss
                # original grads are == loss as matrix
                # original -R come from networks mistakes or:
                # from negative futures ( rewards-rewards_mean )
                # from negative advantage ( new+R-old=error )

                grads = tape.gradient ( loss , [ self.network.model.trainable_variables , std ] )
                grad1 = [ tf.clip_by_norm ( t = w , clip_norm = 5.0 ) for w in   grads [ 0 ]   ] # alternative to TAU in DQN
                grad2 = [ tf.clip_by_norm ( t = w , clip_norm = 5.0 ) for w in [ grads [ 1 ] ] ] # alternative to TAU in DQN

                self.network.optimizer.apply_gradients ( zip ( grad1 , self.network.model.trainable_variables ) )
                self.network.optimizer.apply_gradients ( zip ( grad2 ,                                [ std ] ) )

                print ( '\r' , 'beta:' , round ( float ( self.settings.beta                    ) , 4 ) ,
                              ' step:' , round (   int ( self.network.lr_schedule.current_step )     ) , '/' , self.network.lr_schedule.max_steps         ,
                              ' perc:' ,           int ( self.network.lr_schedule.current_step            /    self.network.lr_schedule.max_steps * 100 ) ,
                              ' rate:' , round ( float ( self.network.lr_schedule.current_rate ) , 7 ) ,
                              ' iter:' , iter_step , '/' , total                                       , end = '' , sep = '' )

        # print ( 'std:' , std ) # != [0,0,0,0]
        self.std = std

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def cross_entropy_loss                        (            self          ) :
        self.replay  = ReplayBuffer               ( settings = self.settings )
        self.network = Network_Input_Value_Action ( settings = self.settings )

        self.std = tf.zeros ( self.settings.action_size , dtype = float )
        self.start = 1

        ie , red_flag = 0 , False
        best_result   =   - np.inf
        scores_window =     deque ( maxlen = 100 )

        for i , w in enumerate ( self.network.model.weights ) : print ( 'el i:' , i , 'shape:' , w.shape )

        weights = [ tf.keras.initializers.Orthogonal ( seed = i ) ( w.shape ) * 1e-3 \
                  \
                    if ( ( i + 1 ) % 2 > 0 ) else np.zeros ( w.shape ) \
                  \
                    for i , w in enumerate ( self.network.model.weights [ -4 : ] ) ]

        self.network.setWeights ( self.network.model.weights [ : -4 ] + weights )

        while not red_flag :
              print ( '' )

              self.play_EP ( ie , self.settings.envs.reset ( train_mode = True ) [ self.settings.brain_name ] )

              if ie > self.start : scores_window.append ( np.mean ( self.collection_rewards_sum ) )
              if ie > self.start : print ( '\nscores_window mean:' , np.mean ( scores_window ) )

              if        ie > self.start and best_result < np.mean ( scores_window ) :
                     if ie > self.start   : best_result = np.mean ( scores_window )

                     self.network.model.save_weights ( 'saved_model_weights_by_tensorflow.h5'                  )
                     print                           ( 'saved_model_weights_by_tensorflow best:' , best_result )
              else : print                           ( '\nbest_scores_window mean:'              , best_result )

              # self.settings.quality -= 0.0005 # if the training has success, the final model quality will be close to 1
              # self.settings.beta = self.settings.beta * 0.995 # block the entropy training if it is reduced to 0
              # does not have sense decreasing beta, because we are considering short batches

              ie = ie + 1
              if   ie >= self.settings.iterations : red_flag = True
