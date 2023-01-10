import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class lambda_function () :
    def __init__ ( self , settings ) : self.settings = settings
    def expand ( self , x ) : return tf.expand_dims ( x , axis = -1 )

# -----------------------------------------------------

class QNetwork_Actor () :

    def __init__ ( self    ,                   settings ) :
        self.settings      =                   settings
        self.object_lambda = lambda_function ( settings )

        self.optimizer = tf.keras.optimizers.Adam ( settings.LR_ACTOR )

        # -------------------------------------------------------------------------------------------
        self.model =     tf.keras.Sequential    ()
        self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'DenseInput_Actor' ) )

        self.model.add ( tf.keras.layers.SimpleRNN ( settings.units1 ,
                                                     activation = tf.nn.relu , name = 'DenseDeep1_Actor' , return_sequences = True  , go_backwards = False ) )

        self.model.add ( tf.keras.layers.SimpleRNN ( settings.units2 ,
                                                     activation = tf.nn.relu , name = 'DenseDeep2_Actor' , return_sequences = False , go_backwards = False ) )

        self.model.add ( tf.keras.layers.Dense ( settings.action_size , activation = tf.nn.tanh , name = 'DenseOtput_Tanh_Actor' ) )
        # -------------------------------------------------------------------------------------------

        self.model.summary ()

    # ----------------------------------------------------------------------

    def predict ( self               , state                                              , verbose = 1       ) :
        return    self.model.predict ( state.reshape ( ( 1 , self.settings.state_size ) ) , verbose = verbose ) [ 0 ]

    def predict_series ( self               , states , verbose = 1       ) :
        return           self.model.predict ( states , verbose = verbose )

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
