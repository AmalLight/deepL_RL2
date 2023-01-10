import tensorflow as tf
import numpy      as np

import tensorflow_probability as tfp

# -----------------------------------------------------

class MyLRSchedule ( tf.keras.optimizers.schedules.LearningRateSchedule ) :

  def __init__ ( self , initial_learning_rate, max_steps ) :
    self.initial_learning_rate = initial_learning_rate
    self.current_rate = initial_learning_rate
    self.current_step = 0
    self.max_steps = max_steps

    self.decay_rate = 1.0

  # Learning rate decay the common method

  def __call__ ( self , step ) :

     # self.current_rate = self.initial_learning_rate * ( 1 / ( 1 + self.decay_rate * ( self.current_step / self.max_steps ) ) )
     # self.current_rate = max ( self.current_rate , 1e-4 )
     return                      self.current_rate

  def addNewSteps ( self , steps ) : self.current_step += steps

# ---------------------------------------------------
# ---------------------------------------------------

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

# ---------------------------------------------------
# ---------------------------------------------------

class Network_Input_Value_Action () :

    def __init__     ( self , settings , RNN = 0 , split = 0 , add_third_layers = 0 ) :
        self.settings       = settings

        self.max_steps = self.settings.max_episodes * self.settings.steps_to_train * self.settings.elite

        self.lr_schedule = MyLRSchedule ( 3e-4 , self.max_steps )

        self.optimizer = tf.keras.optimizers.Adam (
                         learning_rate = self.lr_schedule , # clipnorm = self.settings.clipnorm ,
                         epsilon       = 1e-5             )

        # default clipnorm = 5
        # https://www.youtube.com/watch?v=_-CZr06R5CQ -> CS 152 NN—17 Gradient Clipping

        self.ortog = tf.keras.initializers.Orthogonal            ( gain   = 1e-3 )
        self.regul = tf.keras.regularizers.OrthogonalRegularizer ( factor = 1e-3 ) # less efficient than Normalization

        # https://www.youtube.com/watch?v=vKBNzM3V-Rc -> Computing inverse matrices
        # https://www.youtube.com/watch?v=F6llLO84ROI -> Orthogonal Matrix, initializers
        # https://www.youtube.com/watch?v=4yNrSg7b4JA -> Orthogonal Vectors, regularizers

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        States_Input  = tf.keras.Input ( shape = ( settings.state_size  , ) )
        Actions_Input = tf.keras.Input ( shape = ( settings.action_size , ) )
        Bools_Input   = tf.keras.Input ( shape = ( 1                    , ) )

        # States_Input_norm  = tf.keras.layers.Normalization ( axis = None ) ( States_Input  )
        # Actions_Input_norm = tf.keras.layers.Normalization ( axis = None ) ( Actions_Input )
        States_Input_norm    = States_Input
        Actions_Input_norm   = Actions_Input

        Input_Lambda_1 = None
        Input_Lambda_2 = None

        DenseDeep11 = None
        DenseDeep21 = None
        DenseDeep12 = None
        DenseDeep22 = None

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        # Normalization helps large ammount of short batches to point +- in the same direction

        if RNN == 1 :

            Input_Lambda_1 = tf.keras.layers.Lambda ( self.object_lambda.expand              ,
                                                      input_shape = [ settings.state_size ]  , name = 'Input_Lambda_1' ) ( States_Input_norm )

            DenseDeep11 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_11' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = True  , go_backwards = False ) ( Input_Lambda_1 )

            DenseDeep21 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_21' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = False , go_backwards = False ) ( DenseDeep11 )

            DenseDeep12 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_12' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = True  , go_backwards = False ) ( Input_Lambda_1 )

            DenseDeep22 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_22' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = False , go_backwards = False ) ( DenseDeep12 )
        else:

            DenseDeep11 = tf.keras.layers.Dense ( settings.hidden         , input_shape = [ settings.state_size ] ,

                                                  activation = tf.nn.relu , name = 'DenseDeep11' ) ( States_Input_norm )

            # DenseDeep11 = tf.keras.layers.Normalization ( axis = None ) ( DenseDeep11 )

            DenseDeep21 = tf.keras.layers.Dense ( settings.hidden         ,
                                                  activation = tf.nn.relu , name = 'DenseDeep21' ) ( DenseDeep11 )

            # DenseDeep21 = tf.keras.layers.Normalization ( axis = None ) ( DenseDeep21 )

            DenseDeep12 = tf.keras.layers.Dense ( settings.hidden         , input_shape = [ settings.state_size ] ,

                                                  activation = tf.nn.relu , name = 'DenseDeep12' ) ( States_Input_norm )

            # DenseDeep12 = tf.keras.layers.Normalization ( axis = None ) ( DenseDeep12 )

            DenseDeep22 = tf.keras.layers.Dense ( settings.hidden         ,
                                                  activation = tf.nn.relu , name = 'DenseDeep22' ) ( DenseDeep12 )

            # DenseDeep22 = tf.keras.layers.Normalization ( axis = None ) ( DenseDeep22 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        last_Values  = DenseDeep21 if split else DenseDeep21
        last_Actions = DenseDeep22 if split else DenseDeep21

        if not split : DenseDeep12 = DenseDeep22 = None

        Dense_Value  = None
        Dense_Action = None

        if   add_third_layers :
             Dense_Values  = tf.keras.layers.Dense ( settings.hidden , activation = tf.nn.relu , name = 'Dense_Values'  ) ( last_Values  )
             Dense_Actions = tf.keras.layers.Dense ( settings.hidden , activation = tf.nn.relu , name = 'Dense_Actions' ) ( last_Actions )
        else:
             Dense_Values  = last_Values
             Dense_Actions = last_Actions

        actions_size = self.settings.action_size
        tangent = tf.nn.tanh

        Out_Values  = tf.keras.layers.Dense ( 1            , activation = None    , name = 'Out_Values'  , kernel_initializer = self.ortog )( Dense_Values  )
        Out_Actions = tf.keras.layers.Dense ( actions_size , activation = tangent , name = 'Out_Actions' , kernel_initializer = self.ortog )( Dense_Actions )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        myLayer = MyNormalLayer ( self.settings )

        Out_Actions_2 , log_prob_sum , entropy_sum = myLayer ( Out_Actions , Actions_Input_norm , Bools_Input )

        self.model = tf.keras.Model ( inputs  = [ States_Input , Actions_Input , Bools_Input                ] ,
                                      outputs = [ Out_Values   , Out_Actions_2 , log_prob_sum , entropy_sum ] )
        self.model.summary ()

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
