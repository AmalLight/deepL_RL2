import tensorflow as tf
import numpy      as np

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

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class Network_Input_Value_Action () :

    def __init__     ( self , settings , RNN = 0 , split = 0 , add_third_layers = 0 ) :
        self.settings       = settings

        self.max_steps = self.settings.max_episodes * self.settings.steps_to_train * self.settings.elite

        self.lr_schedule = MyLRSchedule ( 3e-4 , self.max_steps )

        self.optimizer = tf.keras.optimizers.Adam ( learning_rate = self.lr_schedule , clipnorm = 5 , epsilon = 1e-5 )

        # ----------------------------------------------------------------------------

        self.ortog = tf.keras.initializers.Orthogonal            ( gain   = 1e-3 )
        self.regul = tf.keras.regularizers.OrthogonalRegularizer ( factor = 1e-3 )

        # https://www.youtube.com/watch?v=vKBNzM3V-Rc -> Computing inverse matrices
        # https://www.youtube.com/watch?v=F6llLO84ROI -> Orthogonal Matrix
        # https://www.youtube.com/watch?v=4yNrSg7b4JA -> Orthogonal Vectors
        #
        # kernel_regularizer = self.regul, all this for the last layer: 512x4,
        # before activation for forward, after for training
        # kernel it is only for the layer's weight

        # ----------------------------------------------------------------------------

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )

        # -------------------------------------------------------------------------------------------
        Input = tf.keras.Input ( shape = ( settings.state_size , ) )
        Input_Lambda = None
        DenseDeep11  = None
        DenseDeep21  = None
        DenseDeep12  = None
        DenseDeep22  = None

        if RNN == 1 :
            Input_Lambda = tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'LambdaInput' ) (Input)

            DenseDeep11 = tf.keras.layers.SimpleRNN ( settings.hidden         , name = 'RNN_relu_11' ,
                                                      activation = tf.nn.relu ,
                                                      return_sequences = True , go_backwards = False ) ( Input_Lambda )

            DenseDeep21 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_21' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = False , go_backwards = False ) ( DenseDeep11 )

            DenseDeep12 = tf.keras.layers.SimpleRNN ( settings.hidden         , name = 'RNN_relu_12' ,
                                                      activation = tf.nn.relu ,
                                                      return_sequences = True , go_backwards = False ) ( Input_Lambda )

            DenseDeep22 = tf.keras.layers.SimpleRNN ( settings.hidden          , name = 'RNN_relu_22' ,
                                                      activation = tf.nn.relu  ,
                                                      return_sequences = False , go_backwards = False ) ( DenseDeep12 )
        else:
            DenseDeep11 = tf.keras.layers.Dense ( settings.hidden                       ,
                                                  input_shape = [ settings.state_size ] ,
                                                  activation = tf.nn.relu               , name = 'DenseDeep11' ) ( Input )

            DenseDeep21 = tf.keras.layers.Dense ( settings.hidden         ,
                                                  activation = tf.nn.relu , name = 'DenseDeep21' ) ( DenseDeep11 )

            DenseDeep12 = tf.keras.layers.Dense ( settings.hidden                       ,
                                                  input_shape = [ settings.state_size ] ,
                                                  activation = tf.nn.relu               , name = 'DenseDeep12' ) ( Input )

            DenseDeep22 = tf.keras.layers.Dense ( settings.hidden         ,
                                                  activation = tf.nn.relu , name = 'DenseDeep22' ) ( DenseDeep12 )

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

        Out_Values  = tf.keras.layers.Dense ( 1                    , activation = None          , name = 'Out_Values'  , use_bias = True ,
                                              kernel_initializer = self.ortog ) ( Dense_Values  )
        Out_Actions = tf.keras.layers.Dense ( settings.action_size , activation = tf.nn.tanh    , name = 'Out_Actions' , use_bias = True ,
                                              kernel_initializer = self.ortog ) ( Dense_Actions )

        self.model = tf.keras.Model ( inputs = [ Input ] , outputs = [ Out_Values , Out_Actions ] )
        # -------------------------------------------------------------------------------------------

        self.model.summary ()

    def setWeights ( self , weights ) : self.model.set_weights ( weights )

