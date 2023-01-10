import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class Network_Critic () :
    def __init__     ( self , settings , RNN_LSTM = 0 ) :
        self.settings       = settings
        self.optimizer      = tf.keras.optimizers.Adam ( learning_rate = 3e-4 , epsilon = 1e-5 )

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )

        # -------------------------------------------------------------------------------------------
        self.model = tf.keras.Sequential ()

        if RNN_LSTM == 1 :
           self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'LambdaInput' ))

           self.model.add ( tf.keras.layers.SimpleRNN ( settings.hidden         ,
                                                        return_sequences = True , go_backwards = False    ,
                                                        name = 'RNN_relu_1'     , activation = tf.nn.relu ))

           self.model.add ( tf.keras.layers.SimpleRNN ( settings.hidden          ,
                                                        return_sequences = False , go_backwards = False    ,
                                                        name = 'RNN_relu_2'      , activation = tf.nn.relu ))
        elif RNN_LSTM == 2 :
           self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'LambdaInput' ))

           self.model.add ( tf.keras.layers.LSTM ( settings.hidden         ,
                                                   return_sequences = True , go_backwards = False    ,
                                                   name = 'LSTM_relu_1'    , activation = tf.nn.relu ))

           self.model.add ( tf.keras.layers.LSTM ( settings.hidden          ,
                                                   return_sequences = False , go_backwards = False    ,
                                                   name = 'LSTM_relu_2'     , activation = tf.nn.relu ))
        else :
           self.model.add ( tf.keras.layers.Dense ( settings.hidden       , input_shape = [ settings.state_size ] ,
                                                    name = 'Dense_relu_1' , activation = tf.nn.relu ))

           self.model.add ( tf.keras.layers.Dense ( settings.hidden       ,
                                                    name = 'Dense_relu_2' , activation = tf.nn.relu ))

        self.model.add ( tf.keras.layers.Dense ( 1 , activation = None , name = 'Dense_Value' ))
        # -------------------------------------------------------------------------------------------

        self.model.compile ( optimizer = self.optimizer , loss = 'mean_squared_error' , metrics = [ 'mse' , 'mae' , 'accuracy' ] )
        self.model.summary ()

    def setWeights     ( self , weights                             ) : self.model.set_weights ( weights )
    def training       ( self , states , labels , verbose = 0       ) :
        self.model.fit (        states , labels , verbose = verbose )

