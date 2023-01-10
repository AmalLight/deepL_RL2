import tensorflow             as tf
import tensorflow_probability as tfp
import numpy                  as np

# pip3           install           tensorflow_probability
# python3 -m pip install --upgrade tensorflow

import random , sys , threading , math , time , gym

from unityagents import UnityEnvironment
from collections import deque

from model_Critic import Network_Critic
from model_Actor  import Network_Actor
from replay       import ReplayBuffer

# ---------------------------------------------------------------------------
# A2C ( sync ) A3C ( async )
# Big Deep Network == dense matrix == short long memory without buffer.
# dense matrix   at all for MC and TD is not possible ( in continuous )
# states,actions at all for MC and TD is not possible ( in continuous )
# is not possible to evaluate infinite actions without a very big buffer ( DDPG )

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
        # RNN and model think about it

        # states /= np.linalg.norm ( states , axis=-1 , keepdims=True ) + self.settings.small_constant

        for t in range ( self.settings.max_try ) :

            # ------------------------------------------------------

            null_actions = tf.zeros ( states.shape [ 0 ] , self.settings.action_size )

            values                            = self.network_critic.model (   states                          ).numpy () # ( 20 , 1 )
            continuosV , old_log_prob_sum , _ = self.network_actor .model ( [ states , null_actions , [ 0 ] ] )

            continuosV       = continuosV      .numpy () # ( 20 , 4 )
            old_log_prob_sum = old_log_prob_sum.numpy () # ( 20 , 1 )

            # ------------------------------------------------------

            envs_info = self.settings.envs.step ( continuosV ) [ self.settings.brain_name ]

            next_states = np.array ( envs_info.vector_observations )
            rewards     = np.array ( envs_info.rewards             )
            dones       = np.array ( envs_info.local_done          )

            # next_states /= np.linalg.norm ( next_states , axis=-1 , keepdims=True ) + self.settings.small_constant 

            # ------------------------------------------------------
            # ------------------------------------------------------

            self.collection_continuosV. append ( continuosV       ) # append on tail
            self.collection_states.     append ( states           ) # append on tail
            self.collection_next_states.append ( next_states      ) # append on tail
            self.collection_dones.      append ( dones            ) # append on tail
            self.collection_values.     append ( values           ) # append on tail
            self.collection_probs.      append ( old_log_prob_sum ) # append on tail

            self.collection_rewards_step += rewards
            self.collection_rewards_sum  += rewards

            # rewards /= np.linalg.norm ( rewards , axis=-1 , keepdims=True ) + self.settings.small_constant

            self.collection_rewards.append ( rewards ) # append on tail

            states = next_states

            print ( '\rEP:' , ie + 1 , 't:' , t + 1 ,
                      'EP:'   , round ( np.mean ( self.collection_rewards_sum  ) , 3 ) ,
                      'Step:' , round ( np.mean ( self.collection_rewards_step ) , 3 ) ,
                      'MEM1:' ,   len (           self.collection_states       )       ,
                      'MEM2:' ,                   self.replay.len_memory      ()       , end='' )

            # ------------------------------------------------------
            # ------------------------------------------------------

            full_memory = ( ( self.replay.len_memory () + (t+1) * self.settings.elite ) == self.settings.memory_deque )

            if ( ( t + 1 ) % self.settings.steps_to_train == 0 ) or full_memory :

               real_steps_to_train = np.array ( self.collection_states ).shape [ 0 ]

               self.states_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size  ) )
               self.next_states_elite = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size  ) )
               self.continuosV_elite  = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.action_size ) )
               self.rewards_elite     = np.zeros ( ( real_steps_to_train , self.settings.elite                             ) )
               self.dones_elite       = np.ones  ( ( real_steps_to_train , self.settings.elite                             ) )
               self.values_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , 1                         ) )
               self.probs_elite       = np.zeros ( ( real_steps_to_train , self.settings.elite , 1                         ) )

               if full_memory : print ( '' )
               self.training_for_elite ( real_steps_to_train , full_memory )
               if full_memory : print ( '' )

               self.collection_states      = []
               self.collection_next_states = []
               self.collection_continuosV  = []
               self.collection_dones       = []
               self.collection_rewards     = []
               self.collection_values      = []
               self.collection_probs       = []

               self.collection_rewards_step = np.zeros ( self.settings.cpus )
               if full_memory: self.replay.destroy ()

            if full_memory : break # any ( dones )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def advantage_rewards ( self , rewards , last_missing_prediction , real_steps_to_train ) :

        # TD_sarsa_next = alpha * ( reward + (gamma * new_state_action * dones) - old_Qvalue )
        # Q [ state ] [ action ] = ( old_Qvalue + sarsa )

        advantage_rewards = np.zeros ( ( real_steps_to_train , self.settings.elite , 1 ) )

        self.dones_elite = np.expand_dims ( self.dones_elite  , 2 )
        rewards          = np.expand_dims (           rewards , 2 )

        # --------------------------------------------------------------

        for i in reversed ( range ( real_steps_to_train ) ) :

            old_TD_state_action = self.values_elite [ i     ]
            new_TD_state_action = self.values_elite [ i + 1 ] \
                                  \
                                  if ( ( (i+1) % self.settings.max_try ) > 0 ) \
                                  \
                                  else last_missing_prediction # recursion

            new_TD_state_action *= 0.99 # (X*N)+(0.99**N*X)

            error   = rewards [ i ] + new_TD_state_action * self.dones_elite [ i ] - old_TD_state_action
            advantage_rewards [ i ] = error

        # for  row in                       range ( advantage_rewards.shape [ 0 ]     ) : advantage_rewards [ row , : ] *= ( 0.95 ** row )
        for    row in            reversed ( range ( advantage_rewards.shape [ 0 ]   ) ) :
           for col in                       range ( advantage_rewards.shape [ 1 ]     ) :
               advantage_rewards [ row , col ] += ( advantage_rewards [ row + 1 , col ] * self.dones_elite [ row , col ] * 0.94 ) \
                                                  \
                                                  if ( ( (row+1) % self.settings.max_try ) > 0 ) else 0 # no sum before if for reversed and dones_elite
        # X+0.95*X
        # X+0.95*X
        # X+0
        #
        # X+(0.95*(X+0.95*X))
        # x+x*0.95 = x( 1 + 0.95 ) = x1.95 => 3 step: x1.95 + x0.95^2 = X*2.85 < X*2.95
        #
        # (X*N)+(0.95**N*X) # from reversed X*N contains futures advantages X
        # (X*N)+(0.95**N*X) versus X*N+1
        #
        # if X+(0.95*(X+0.95*(X+0.95*X))) :
        # X + x0.95 + 0.95^2*( X+0.95*X ) ,
        # x1.95 + 0.95^2*x + 0.95^3*x
        #
        # 9*100                = 900
        # (9*99)+(0.95**100*9) = 891
        # if N=100 e x=99: 9900 vs 9801

        # --------------------------------------------------------------

        advantage_rewards = self.myFlatten ( advantage_rewards )

        rewards_mean = np.mean ( advantage_rewards , axis = 0 , keepdims = False )
        rewards_std  = np.std  ( advantage_rewards , axis = 0 , keepdims = False ) # + self.settings.small_constant

        # rewards_std = np.array ( [   1.0   if           r == 0.0 else r for r in rewards_std ] ) # if axis = 0: 1,self.settings.elite
        # rewards_std =                1.0   if rewards_std == 0.0 else            rewards_std     # if flatten:  1,1
        # rewards_std = np.array ( [ [ 1.0 ] if           r == 0.0 else r for r in rewards_std ] ) # if axis = 1: real_steps_to_train,1

        advantage_rewards -= rewards_mean
        advantage_rewards /= rewards_std
        return advantage_rewards

    def future_rewards ( self , rewards , last_missing_prediction ) :

        for    row in  reversed ( range ( rewards.shape [  0  ]   ) ) :
           for col in             range ( rewards.shape [  1  ]     ) :
               rewards_next           = ( rewards  [ row + 1  , col ] * self.dones_elite [ row , col ] ) \
                                        \
                                        if ( ( (row+1) % self.settings.max_try ) > 0 ) \
                                        \
                                        else ( last_missing_prediction [ col ] * self.dones_elite [ row , col ] )

               rewards_next *= 0.99 # (X*N)+(0.99**N*X)
               rewards [ row , col ] += rewards_next

        # future rewards + structural recursion , no sum before if for reversed and dones_elite
        return rewards # no norm here because it is not the advantage

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def myFlatten ( self , values ) :
        da_ritorno = []
        for        i1 in range ( values.shape [ 0 ] )     :
           if              len ( values.shape       ) > 2 :
               for i2 in range ( values.shape [ 1 ] )     : da_ritorno += [ values [ i1 , i2 , : ] ]
           else                                           : da_ritorno += [ values [ i1 ,      : ] ]

        da_ritorno                                   = np.array       ( da_ritorno     )
        if not len ( values.shape ) > 2 : da_ritorno = np.expand_dims ( da_ritorno , 2 )
        return                            da_ritorno

    def training_for_elite ( self , real_steps_to_train , to_train = False ) :

        elite_idxs = self.collection_rewards_step.argsort () [ - self.settings.elite : ] # it Returns max values => direct progress

        for i , e in enumerate ( elite_idxs ) :

            self.states_elite      [ : , i ] = np.array ( self.collection_states      ) [ : , e ]
            self.next_states_elite [ : , i ] = np.array ( self.collection_next_states ) [ : , e ]
            self.continuosV_elite  [ : , i ] = np.array ( self.collection_continuosV  ) [ : , e ]
            self.rewards_elite     [ : , i ] = np.array ( self.collection_rewards     ) [ : , e ]
            self.dones_elite       [ : , i ] = np.array ( self.collection_dones       ) [ : , e ] - 1.0
            self.values_elite      [ : , i ] = np.array ( self.collection_values      ) [ : , e ]
            self.probs_elite       [ : , i ] = np.array ( self.collection_probs       ) [ : , e ]

        # --------------------------------------------------------------

        # last_missing_prediction = np.zeros ( ( self.next_states_elite [ -1 ].shape [ 0 ] , 1 ) ) # no structural recursion

        last_missing_prediction = self.network_critic.model ( self.next_states_elite [ -1 ] ).numpy ()

        self.future_rewards_elite    = self.future_rewards    ( self.rewards_elite , last_missing_prediction                       )
        self.advantage_rewards_elite = self.advantage_rewards ( self.rewards_elite , last_missing_prediction , real_steps_to_train )

        self.training ( self.myFlatten ( np.expand_dims ( self.future_rewards_elite , 2 ) ) , real_steps_to_train , to_train )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training ( self , future_rewards , real_steps_to_train , to_train = False ) :

        self.collection_states     = self.myFlatten ( self.states_elite     )
        self.collection_values     = self.myFlatten ( self.values_elite     )
        self.collection_probs      = self.myFlatten ( self.probs_elite      )
        self.collection_continuosV = self.myFlatten ( self.continuosV_elite )

        # https://github.com/jknthn/reacher-ppo/blob/master/agent.py
        # https://github.com/chris838/reacher/blob/master/ppo_agent.py
        # https://keras.io/examples/rl/ppo_cartpole/

        # from actions to probs, but not dense as MC ( i am using Gaussian distribution )
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal
        # distributions Normal is == np.random.normal

        # ------------------------------------------------------------------------------------------

        for i in range ( real_steps_to_train * self.settings.elite ) :

            self.replay.add ( self.collection_states       [ i ] ,
                              self.collection_continuosV   [ i ] ,
                                   future_rewards          [ i ] ,
                              self.collection_probs        [ i ] ,
                              self.advantage_rewards_elite [ i ] ,
                              self.collection_values       [ i ] ) # append on tail

        total = self.settings.verbosity * self.settings.batch_size
        # self.network_actor.lr_schedule.addNewSteps ( real_steps_to_train * self.settings.elite ) # learning rate decay

        if not to_train : return 0

        # batches are a cheat to use a big hidden size on old computers,
        # and to have different exploration paths

        iter_step = 0 ; print ( '\r' , iter_step , '/' , total , end='' , sep='' )

        # ------------------------------------------------------------------------------------------

        break_actions = False

        for v in range ( self.settings.verbosity ) :

            # More big is self.settings.verbosity more the entropy will be larger. So the loss will be smaller

            self.replay.reset_memory_i ()
            self.replay.shuffle_simple ()
            # self.replay.shuffle      ()

            for i in range ( self.settings.batch_size ) :
                iter_step += 1

                stats , acts , futurs , probs , advs = self.replay.sample ()

                with tf.GradientTape ( persistent = False ) as tape_values :
                     tape_values.reset ()

                     value_predictions = self.network_critic.model ( stats )

                     loss_values = tf.math.reduce_mean ( tf.math.pow ( futurs - value_predictions , 2 ) ) * 0.5

                with tf.GradientTape ( persistent = False ) as tape_actions :
                     tape_actions.reset ()

                     _ , new_log_prob_sum , entropy_sum = self.network_actor.model ( [ stats , acts , [ 1 ] ] )

                     # old_log_prob_sum = tf.math.exp (                            probs    )
                     # new_log_prob_sum = tf.math.exp ( new_log_prob_sum                    ) # log_prob must be log based on 2 or e
                     # ratio            =             ( new_log_prob_sum / old_log_prob_sum ) # >= 0
                     ratio              = tf.math.exp ( new_log_prob_sum -         probs    ) # >= 0
                     # ratio_min = ratio

                     # tf.clip_by_value: max using clip_value_min -> min expolration
                     #                   min using clip_value_max -> min exploitation ( against excessive expolration )
                     #
                     # (ae) ratio * advs: anti-expolration:
                     # if advs > 0: [ae,clip_value_max] | ae < clip_value_min else advs < 0 [clip_value_min,clip_value_max]

                     ratio_min = tf.math.reduce_min   (
                                 tf.convert_to_tensor ( [ ratio                                                  * advs , \
                                 tf.clip_by_value     (   ratio , clip_value_min = 1 - self.settings.clip_size ,
                                                                  clip_value_max = 1 + self.settings.clip_size ) * advs ] ) , axis = 0 )
                     # print ( ratio_min.shape == ratio.shape )

                     loss_actions = - tf.math.reduce_mean ( ratio_min ) # + self.settings.beta * entropy_sum )

                     # mean(a) + mean(b) = mean( a + b )
                     #
                     # a = 3,9 ; b = 6,4
                     # first: 12/2 + 10/2         = 11
                     # secon: (9   + 13)/2 = 22/2 = 11
                     #
                     # entropy is opposite to ratio -> new = old_probs by new_probs instead of new = new_probs / old_probs
                     # entropy without -entr is: -p(x) * log(x)
                     # entropy punishs prob% close to the median value ( watch normal_distribution.jpg for it ) and favours lower prob%
                     #
                     # old_probs is the potential
                     # entropy   is the distance == log expolration ( i think - is automatic )
                     # advs      is the error

                # watch original -R as +Loss
                # grads are == loss as matrix
                # original -R come from networks mistakes or:
                # from negative advantage ( R-mean/std      )
                # from negative advantage ( new+R-old=error )

                if not break_actions :

                   # mean sum : 3,6 && 2,5 => 4.5 , 3.5 => 8
                   # sum mean : 3,6 && 2,5 =>   9 , 7   => 8

                   quality_actions = tf.math.reduce_mean ( probs - new_log_prob_sum )

                   if quality_actions <= 1.5 * 0.01 :                              self.network_actor.optimizer.minimize         \
                                                       ( loss_actions , var_list = self.network_actor .model.trainable_variables , tape = tape_actions )
                self.network_critic.optimizer.minimize ( loss_values  , var_list = self.network_critic.model.trainable_variables , tape = tape_values  )

                print ( '\r' , 'beta:' , round ( float ( self.settings.beta      ) , 7 ) ,
                              ' MEM2:' ,                 self.replay.len_memory () ,
                              ' iter:' , iter_step , '/' , total                   , end = '' , sep = '' )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def play_round ( self , envs_info = None ) :

        states = envs_info.vector_observations
        scores = np.zeros ( self.settings.cpus )

        while True :

              null_actions = tf.zeros ( states.shape [ 0 ] , self.settings.action_size )

              actions , _ , _ = self.network_actor.model ( [ states , null_actions , [ 0 ] ] )
              actions = actions.numpy ()

              env_info = self.settings.envs.step ( actions )[ self.settings.brain_name ]

              next_states = env_info.vector_observations
              rewards     = env_info.rewards
              dones       = env_info.local_done
              scores     += env_info.rewards

              states = next_states

              if np.any  (  dones ) : break
        return   np.mean ( scores )

    def cross_entropy_loss ( self ) :

        self.replay         = ReplayBuffer   ( settings = self.settings           )
        self.network_critic = Network_Critic ( settings = self.settings , RNN = 0 )
        self.network_actor  = Network_Actor  ( settings = self.settings , RNN = 0 )

        # usually RNN is used when we have to find a sequence that helps to predict better a state

        self.start = -1

        ie , red_flag  = 0 , False
        best_result    =   - np.inf
        scores_window  =     deque ( maxlen = 100 )

        for i , w in enumerate ( self.network_critic.model.weights ) : print ( 'el i:' , i , 'shape critic:' , w.shape )
        for i , w in enumerate ( self.network_actor .model.weights ) : print ( 'el i:' , i , 'shape actor:'  , w.shape )

        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------

        # Compute the qr factorization of a matrix. Factor the matrix a as qr, where q is orthonormal and r is upper-triangular.
        # https://www.youtube.com/watch?v=FAnNBw7d0vg -> QR decomposition (for square matrices)

        # q_weights = []
        #
        # for w in self.network_actor.model.weights [ -1 : ] :
        #     print (w.shape)
        #     if     len ( w.shape ) > 1 :
        #            q , r = np.linalg.qr ( w )
        #            q_weights += [q * 1e-3] # w_scale from layer_init in utils
        #     else : q_weights += [np.zeros (w.shape)] # use_bias=False
        #
        # self.network.setWeights ( self.network_actor.model.weights [ : -1 ] + q_weights )

        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------

        while not red_flag :
              print ( '' )

              self.play_EP              ( ie , self.settings.envs.reset ( train_mode = True ) [ self.settings.brain_name ] )
              np_mean = self.play_round (      self.settings.envs.reset ( train_mode = True ) [ self.settings.brain_name ] )

              if ie > self.start : scores_window.append ( np_mean ) # np.mean ( self.collection_rewards_sum )
              if ie > self.start : print ( '\nscores_window mean:' , np.mean ( scores_window ) )

              if        ie > self.start and best_result < np.mean ( scores_window ) :
                     if ie > self.start   : best_result = np.mean ( scores_window )

                     self.network_critic.model.save_weights ( 'saved_critic_model_weights_by_tensorflow.h5' )
                     self.network_actor .model.save_weights ( 'saved_actor__model_weights_by_tensorflow.h5' )

                     print ( 'saved_models_weights_by_tensorflow best:' , best_result )
              else : print ( 'best_scores_window mean:'                 , best_result )

              # self.settings.beta = max ( self.settings.beta * 0.995 , 0.0001 ) # block the entropy if it is reduced to 0

              ie = ie + 1
