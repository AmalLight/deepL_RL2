import numpy as np
import torch

import random , sys , threading , math , time , gym

from unityagents import UnityEnvironment
from collections import deque

from model_torch  import PPOPolicyNetwork
from import_utils import Batcher

# ---------------------------------------------------------------------------

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

        for t in range ( self.settings.max_try ) :

            # ------------------------------------------------------

            continuosV , old_log_prob_sum , _ , values = self.network ( states )

            continuosV       = continuosV      .cpu ().detach ().numpy ()
            old_log_prob_sum = old_log_prob_sum.cpu ().detach ().numpy ()
            values           = values          .cpu ().detach ().numpy ()

            # ------------------------------------------------------

            envs_info = self.settings.envs.step ( continuosV ) [ self.settings.brain_name ]

            next_states = np.array ( envs_info.vector_observations )
            rewards     = np.array ( envs_info.rewards             )
            dones       = np.array ( envs_info.local_done          )

            # ------------------------------------------------------

            self.collection_continuosV. append ( continuosV       ) # append on tail
            self.collection_states.     append ( states           ) # append on tail
            self.collection_next_states.append ( next_states      ) # append on tail
            self.collection_dones.      append ( dones            ) # append on tail
            self.collection_values.     append ( values           ) # append on tail
            self.collection_probs.      append ( old_log_prob_sum ) # append on tail

            self.collection_rewards_step  += rewards
            self.collection_rewards_sum   += rewards
            self.collection_rewards.append ( rewards ) # append on tail

            states = next_states

            print ( '\rEP:' , ie + 1 , 't:' , t + 1 ,
                      'EP:'   , round ( np.mean ( self.collection_rewards_sum  ) , 3 ) ,
                      'Step:' , round ( np.mean ( self.collection_rewards_step ) , 3 ) ,
                      'MEM1:' ,   len (           self.collection_states       )       , end='' )

            # ------------------------------------------------------

            full_memory = ( ( np.array ( self.collection_states ).shape [ 0 ] * self.settings.elite ) == self.settings.memory_deque )

            if ( ( t + 1 ) % self.settings.steps_to_train == 0 ) or full_memory :

               real_steps_to_train = np.array ( self.collection_states ).shape [ 0 ]

               self.states_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size   ) )
               self.next_states_elite = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.state_size   ) )
               self.continuosV_elite  = np.zeros ( ( real_steps_to_train , self.settings.elite , self.settings.action_size  ) )
               self.rewards_elite     = np.zeros ( ( real_steps_to_train , self.settings.elite                              ) )
               self.dones_elite       = np.ones  ( ( real_steps_to_train , self.settings.elite                              ) )
               self.values_elite      = np.zeros ( ( real_steps_to_train , self.settings.elite , 1                          ) )
               self.probs_elite       = np.zeros ( ( real_steps_to_train , self.settings.elite , 1                          ) )

               if full_memory : print ( '' )
               self.training_for_elite ( real_steps_to_train , full_memory )
               if full_memory : print ( '' )

               self.collection_states       = []
               self.collection_next_states  = []
               self.collection_continuosV   = []
               self.collection_dones        = []
               self.collection_rewards      = []
               self.collection_values       = []
               self.collection_probs        = []
               self.collection_rewards_step = np.zeros ( self.settings.cpus )

            if full_memory : break

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def advantage_rewards ( self , rewards , last_missing_prediction , real_steps_to_train ) :

        advantage_rewards = np.zeros ( ( real_steps_to_train , self.settings.elite , 1 ) )

        for row in reversed ( range ( real_steps_to_train ) ) :

            old_TD_state_action = self.values_elite [ row     ]
            new_TD_state_action = self.values_elite [ row + 1 ] \
                                  \
                                  if ( ( (row+1) % self.settings.max_try ) > 0 ) \
                                  \
                                  else last_missing_prediction # recursion

            new_TD_state_action *= ( self.dones_elite [ row ] * 0.99 )

            advantage_rewards [ row ]  = rewards [ row ] + new_TD_state_action - old_TD_state_action
            advantage_rewards [ row ] += ( advantage_rewards [ row + 1 ] * self.dones_elite [ row ] * 0.94 ) \
                                         \
                                         if ( ( (row+1) % self.settings.max_try ) > 0 ) else 0

        advantage_rewards = self.myFlatten ( advantage_rewards )

        rewards_mean = np.mean ( advantage_rewards , axis = 0 , keepdims = False )
        rewards_std  = np.std  ( advantage_rewards , axis = 0 , keepdims = False ) # + self.settings.small_constant

        advantage_rewards -= rewards_mean
        advantage_rewards /= rewards_std
        return advantage_rewards

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def future_rewards ( self , rewards , last_missing_prediction ) :

        self.dones_elite = np.expand_dims ( self.dones_elite , 2 )

        for row in  reversed ( range ( rewards.shape [  0 ] ) ) :
            rewards_next           = ( rewards  [ row + 1 ] ) \
                                     \
                                     if ( ( (row+1) % self.settings.max_try ) > 0 ) \
                                     \
                                     else ( last_missing_prediction )

            rewards [ row ] += ( rewards_next * self.dones_elite [ row ] * 0.99 )

        return rewards

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def myFlatten ( self , values ) :
        da_ritorno = []
        for      i1 in range ( values.shape [ 0 ] )     :
           if            len ( values.shape       ) > 2 :
             for i2 in range ( values.shape [ 1 ] )     : da_ritorno += [ values [ i1 , i2 , : ] ]

        da_ritorno                                   = np.array       ( da_ritorno     )
        if not len ( values.shape ) > 2 : da_ritorno = np.expand_dims ( da_ritorno , 2 )
        return                            da_ritorno

    def training_for_elite ( self , real_steps_to_train , to_train = False ) :

        elite_idxs = range ( self.settings.elite )

        for i , e in enumerate ( elite_idxs ) :

            self.states_elite      [ : , i ] = np.array ( self.collection_states      ) [ : , e ]
            self.next_states_elite [ : , i ] = np.array ( self.collection_next_states ) [ : , e ]
            self.continuosV_elite  [ : , i ] = np.array ( self.collection_continuosV  ) [ : , e ]
            self.rewards_elite     [ : , i ] = np.array ( self.collection_rewards     ) [ : , e ]
            self.dones_elite       [ : , i ] = np.array ( self.collection_dones       ) [ : , e ] - 1.0
            self.values_elite      [ : , i ] = np.array ( self.collection_values      ) [ : , e ]
            self.probs_elite       [ : , i ] = np.array ( self.collection_probs       ) [ : , e ]

        # --------------------------------------------------------------

        _ , _ , _ , last_missing_prediction = self.network ( self.next_states_elite [ -1 ] )
        last_missing_prediction = last_missing_prediction.cpu ().detach ().numpy ()

        self.rewards_elite = np.expand_dims ( self.rewards_elite , 2 )

        self.future_rewards_elite    = self.future_rewards    ( self.rewards_elite.copy () , last_missing_prediction                       )
        self.advantage_rewards_elite = self.advantage_rewards ( self.rewards_elite.copy () , last_missing_prediction , real_steps_to_train )

        self.training ( self.myFlatten ( self.future_rewards_elite ) , real_steps_to_train , to_train )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def thread_batch_optimizer ( self , future_rewards , iter_step , total , batch_indices ) :

        stats  = self.collection_states       [ batch_indices ]
        acts   = self.collection_continuosV   [ batch_indices ]
        futurs =      future_rewards          [ batch_indices ]
        probs  = self.collection_probs        [ batch_indices ]
        advs   = self.advantage_rewards_elite [ batch_indices ]

        _ , new_log_prob_sum , entropy_loss , values = self.network ( stats , acts )

        ratio = ( new_log_prob_sum - probs ).exp ()

        policy_loss  = - torch.min ( ratio                                                                     * advs ,
                                     ratio.clamp ( 1 - self.settings.clip_size , 1 + self.settings.clip_size ) * advs ).mean ( 0 ) \
                       \
                       - self.settings.beta * entropy_loss.mean ()

        value_loss = 0.5 * ( futurs - values ).pow ( 2 ).mean ()

        self.optim.zero_grad ()
        ( policy_loss + value_loss ).backward ()
        torch.nn.utils.clip_grad_norm_ ( self.network.parameters () , 5 )
        self.optim.step ()

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training ( self , future_rewards , real_steps_to_train , to_train = False ) :

        self.collection_states     = self.myFlatten ( self.states_elite     )
        self.collection_values     = self.myFlatten ( self.values_elite     )
        self.collection_probs      = self.myFlatten ( self.probs_elite      )
        self.collection_continuosV = self.myFlatten ( self.continuosV_elite )

        batcher = Batcher (             self.collection_states.shape [ 0 ] // self.settings.batch_size ,
                          [ np.arange ( self.collection_states.shape [ 0 ] ) ] )

        self.collection_states       = torch.Tensor ( self.collection_states       )
        self.collection_continuosV   = torch.Tensor ( self.collection_continuosV   )
        future_rewards               = torch.Tensor (      future_rewards          )
        self.collection_probs        = torch.Tensor ( self.collection_probs        )
        self.advantage_rewards_elite = torch.Tensor ( self.advantage_rewards_elite )

        # ------------------------------------------------------------------------------------------

        total = self.settings.verbosity * self.settings.batch_size

        if not to_train : return 0
        iter_step = 0   ; print ( '\r' , iter_step , '/' , total , end='' , sep='' )

        # ------------------------------------------------------------------------------------------

        batcher.shuffle ()

        for v in range ( self.settings.verbosity ) :

            while not batcher.end () :

                iter_step += 1

                batch_indices = batcher.next_batch () [ 0 ]
                batch_indices = torch.Tensor ( batch_indices ).long ()

                self.thread_batch_optimizer ( future_rewards , iter_step , total , batch_indices )

            print ( '\r' , 'beta:' , round ( float ( self.settings.beta ) , 7                   ) ,
                          ' iter:' , iter_step , '/' , total              , end = '' , sep = '' )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def play_round ( self , envs_info = None ) :

        states = envs_info.vector_observations
        scores = np.zeros ( self.settings.cpus )

        while True :

              actions , _ , _ , _ = self.network ( states )
              actions = actions.cpu ().detach ().numpy ()

              env_info = self.settings.envs.step ( actions )[ self.settings.brain_name ]

              next_states = env_info.vector_observations
              rewards     = env_info.rewards
              dones       = env_info.local_done
              scores     += env_info.rewards

              states = next_states

              if np.any  (  dones ) : break
        return   np.mean ( scores )

    def cross_entropy_loss ( self ) :

        self.network = PPOPolicyNetwork ( self.settings.state_size , self.settings.action_size , self.settings.hidden )

        self.optim = torch.optim.Adam ( self.network.parameters () , 3e-4 , eps = 1e-5 )
        self.start = -1

        ie , red_flag  = 0 , False
        best_result    =   - np.inf
        scores_window  =     deque ( maxlen = 100 )

        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------

        while not red_flag :
              print ( '' )

              self.play_EP              ( ie , self.settings.envs.reset ( train_mode = True ) [ self.settings.brain_name ] )
              np_mean = self.play_round (      self.settings.envs.reset ( train_mode = True ) [ self.settings.brain_name ] )

              if ie > self.start : scores_window.append ( np_mean )
              if ie > self.start : print ( '\nscores_window mean:' , np.mean ( scores_window ) )

              if        ie > self.start and best_result < np.mean ( scores_window ) :
                     if ie > self.start   : best_result = np.mean ( scores_window )

                     torch.save ( self.network.state_dict () , 'saved_models_weights_by_torch.pth' )

                     print ( 'saved_models_weights_by_torch best:' , best_result )
              else : print ( 'best_scores_window mean:'            , best_result )

              ie = ie + 1
