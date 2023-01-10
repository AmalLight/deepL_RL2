import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class MyLRSchedule ( tf.keras.optimizers.schedules.LearningRateSchedule ) :

  def __init__ ( self , initial_learning_rate, max_steps ) :
    self.initial_learning_rate = initial_learning_rate
    self.current_rate = initial_learning_rate
    self.current_step = 0
    self.max_steps = max_steps

  def __call__ ( self , step ) :
     # self.current_step = step
     self.current_rate = self.initial_learning_rate * ( 1 - ( self.current_step / self.max_steps ) )
     return self.current_rate

  def addNewSteps ( self , steps ) : self.current_step += steps

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class Network_Input_Value_Action () :

    def __init__     ( self , settings ) :
        self.settings       = settings

        self.max_steps = self.settings.max_episodes * self.settings.memory_deque
        # X=200 ep, Y time of steps_to_train, Z=loading of steps_to_train*elite

        self.lr_schedule = MyLRSchedule ( 3e-4 , self.max_steps )

        self.optimizer = tf.keras.optimizers.Adam ( learning_rate = self.lr_schedule , epsilon = 1e-5 )

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )

        # -------------------------------------------------------------------------------------------
        Input      = tf.keras.Input ( shape = ( None , settings.state_size , ) )
        DenseDeep1 = None
        DenseDeep2 = None

        DenseDeep1 = tf.keras.layers.Dense ( settings.hidden         ,
                                             activation = tf.nn.relu , name = 'DenseDeep1' ) ( Input )

        DenseDeep2 = tf.keras.layers.Dense ( settings.hidden         ,
                                             activation = tf.nn.relu , name = 'DenseDeep2' ) ( DenseDeep1 )

        Out1_Value  = tf.keras.layers.Dense ( 1                    , activation = None       , name = 'Out1_Value'  ) ( DenseDeep2 )
        Out2_Action = tf.keras.layers.Dense ( settings.action_size , activation = tf.nn.tanh , name = 'Out2_Action' ) ( DenseDeep2 )

        self.model = tf.keras.Model ( inputs = [ Input ] , outputs = [ Out1_Value , Out2_Action ] )
        # -------------------------------------------------------------------------------------------

        self.model.summary ()

    def setWeights ( self , weights ) : self.model.set_weights ( weights )

