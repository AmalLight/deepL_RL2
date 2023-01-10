import tensorflow as tf
import numpy      as np

import tensorflow_probability as tfp

# -----------------------------------------------------

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class MyNormalLayer ( tf.keras.layers.Layer ) :

    def __init__ ( self , settings ) :

        super ( MyNormalLayer , self ).__init__ ()

        self.settings = settings

    def build                  ( self , input_shape                          ) :
        ones     = tf.ones     ( ( 1  , input_shape [ -1 ] ) , dtype = float )
        self.std = tf.Variable ( ones , name = 'std'         , dtype = float )

    def call ( self , inputs , actions , bool_actions ) :

        # tfp_dist = tfp.distributions.Normal ( loc = inputs , scale = tf.math.softplus ( self.std ) )

        tfp_dist = tfp.distributions.Normal ( loc = inputs , scale = self.std )

        if bool_actions == 0 : actions = tfp_dist.sample ()

        log_prob = tfp_dist.log_prob ( actions )

        log_prob_sum = tf.math.reduce_sum ( log_prob , axis = 1 , keepdims = True )

        entropy = tfp_dist.entropy ()

        entropy_sum = tf.math.reduce_sum ( entropy , axis = 1 , keepdims = False )

        # print ( self.std          ) # != [1,1,1,1]
        # print ( bool_actions < -1 ) # != True/False

        return actions , log_prob_sum , entropy_sum

# -----------------------------------------------------
# -----------------------------------------------------

class Network_Actor () :
    def __init__    ( self , settings , RNN = 0 ) :
        self.settings      = settings
        self.optimizer     = tf.keras.optimizers.Adam ( learning_rate = 3e-4 , epsilon = 1e-5 )

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )

        self.ortog = tf.keras.initializers.Orthogonal ( gain = 1e-3 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        States_Input  = tf.keras.Input ( shape = ( settings.state_size  , ) )
        Actions_Input = tf.keras.Input ( shape = ( settings.action_size , ) )
        Bools_Input   = tf.keras.Input ( shape = ( 1                    , ) )

        Input_Lambda_1  = None
        Input_Reduced   = None
        Input_Reduced_2 = None

        DenseDeep1 = None
        DenseDeep2 = None

        if self.settings.conv_kernel > 0 :

          Input_Lambda_1  = tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'Input_Lambda_1' ) ( States_Input )

          Input_Reduced   = tf.keras.layers.Conv1D ( 4  , kernel_size = self.settings.conv_kernel , activation = self.settings.basic , name = 'Input_Reduced'   ) \
                                                   ( Input_Lambda_1 )

          Input_Reduced_2 = tf.keras.layers.Conv1D ( 16 , kernel_size = self.settings.conv_kernel , activation = self.settings.basic , name = 'Input_Reduced_2' ) \
                                                   ( Input_Reduced  )

        elif RNN == 1 :

          Input_Reduced_2 = tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'Input_Lambda_1' ) \
                                                   ( States_Input )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        if RNN == 1 :

           DenseDeep1 = tf.keras.layers.SimpleRNN ( settings.hidden                  , name = 'RNN_relu_1' ,
                                                    activation = self.settings.basic ,
                                                    return_sequences = True          , go_backwards = False ) ( Input_Reduced_2 )

           DenseDeep2 = tf.keras.layers.SimpleRNN ( settings.hidden                  , name = 'RNN_relu_2' ,
                                                    activation = self.settings.basic ,
                                                    return_sequences = False         , go_backwards = False ) ( DenseDeep1 )
        else:

           if self.settings.conv_kernel > 0 : Reduced_Flatten = tf.keras.layers.Flatten ( name = 'Flatten' ) ( Input_Reduced_2 )

           DenseDeep1 = tf.keras.layers.Dense ( self.settings.hidden , name = 'DenseDeep1' , activation = self.settings.basic ) \
                                              \
                                              ( Reduced_Flatten if self.settings.conv_kernel > 0 else States_Input )

           if self.settings.conv_kernel == 0 :
                  DenseDeep2 = tf.keras.layers.Dense ( self.settings.hidden , name = 'DenseDeep2' , activation = self.settings.basic ) ( DenseDeep1 )
           else : DenseDeep2 = DenseDeep1

        actions_size = self.settings.action_size

        Out_Actions = tf.keras.layers.Dense ( actions_size , activation = self.settings.final , name = 'Out_Actions' , kernel_initializer = self.ortog ) \
                                            ( DenseDeep2 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        myLayer = MyNormalLayer ( self.settings )

        Out_Actions_2 , log_prob_sum , entropy_sum = myLayer ( Out_Actions , Actions_Input , Bools_Input )

        self.model = tf.keras.Model ( inputs  = [ States_Input , Actions_Input , Bools_Input ] ,
                                      outputs = [ Out_Actions_2 , log_prob_sum , entropy_sum ] )
        self.model.summary ()

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
