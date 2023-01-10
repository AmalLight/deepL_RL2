import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class lambda_function () :
    def __init__ ( self , settings ) : self.settings = settings
    def expand ( self , x ) : return tf.expand_dims ( x , axis = -1 )

# -----------------------------------------------------

class QNetwork_Critic () :

    def __init__ ( self    ,                   settings ) :
        self.settings      =                   settings
        self.object_lambda = lambda_function ( settings )

        self.optimizer = tf.keras.optimizers.Adam ( settings.LR_CRITIC )

        # -------------------------------------------------------------------------------------------
        Input1 = tf.keras.Input ( shape = ( settings.state_size  , ) )
        Input2 = tf.keras.Input ( shape = ( settings.action_size , ) )

        Input1_expand = tf.keras.layers.Lambda ( self.object_lambda.expand , name = 'DenseExpand_Critic_Critic' ) ( Input1 )
        Input2_expand = tf.keras.layers.Lambda ( self.object_lambda.expand , name = 'DenseExpand_Critic_Actor'  ) ( Input2 )

        State_out = tf.keras.layers.SimpleRNN (  settings.units1         ,
                                                 activation = tf.nn.relu , 
                                                 return_sequences = True , go_backwards = False , name = 'DenseDeep1_Critic_Critic' ) ( Input1_expand )

        Action_out = tf.keras.layers.SimpleRNN ( settings.units1         ,
                                                 activation = tf.nn.relu , 
                                                 return_sequences = True , go_backwards = False , name = 'DenseDeep1_Critic_Actor'  ) ( Input2_expand )

        Input3 = tf.keras.layers.Concatenate ( axis = 1 ) ( [ State_out , Action_out ] )

        Output3 = tf.keras.layers.SimpleRNN (    settings.units2          ,
                                                 activation = tf.nn.relu  , 
                                                 return_sequences = False , go_backwards = False , name = 'DenseDeep2_Critic' ) ( Input3  )
        Output4 = tf.keras.layers.Dense     ( 1                           , activation   = None  , name = 'DenseOtput_Critic' ) ( Output3 )
        # -------------------------------------------------------------------------------------------

        self.model = tf.keras.Model ( [ Input1 , Input2 ] , Output4 )
        self.model.compile ( optimizer = self.optimizer , loss = 'mean_squared_error' , metrics = [ 'mse' , 'mae' , 'accuracy' ] )
        self.model.summary ()

    # ----------------------------------------------------------------------

    def training ( self           , states , labels , verbose = 1       ) :
        result   = self.model.fit ( states , labels , verbose = verbose )

        if verbose : print ( 'QNetwork Critic local loss:' , result.history [ 'loss'     ][ -1 ] ,
                                               'accuracy:' , result.history [ 'accuracy' ][ -1 ] )

    def predict_series ( self               , states , verbose = 1       ) :
        return           self.model.predict ( states , verbose = verbose )

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
