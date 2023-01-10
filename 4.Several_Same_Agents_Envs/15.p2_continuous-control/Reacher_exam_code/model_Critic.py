import tensorflow as tf
import numpy      as np

# -----------------------------------------------------

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class Network_Critic () :
    def __init__     ( self , settings , RNN = 0 ) :
        self.settings       = settings
        self.optimizer      = tf.keras.optimizers.Adam ( learning_rate = 1e-4 , epsilon = 1e-5 )

        self.input_shape = [ settings.state_size ]
        print ( 'input_shape:' , self.input_shape )

        self.object_lambda = lambda_function ( settings )
 
        self.ortog = tf.keras.initializers.Orthogonal ( gain = 1e-3 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        self.model = tf.keras.Sequential ()

        if self.settings.conv_kernel > 0 :

          self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'Input_Lambda_1' ) )

          self.model.add ( tf.keras.layers.Conv1D ( 4  , kernel_size = self.settings.conv_kernel , activation = self.settings.basic , name = 'Input_Reduced'   ) )
          self.model.add ( tf.keras.layers.Conv1D ( 16 , kernel_size = self.settings.conv_kernel , activation = self.settings.basic , name = 'Input_Reduced_2' ) )

        elif RNN == 1 :

          self.model.add ( tf.keras.layers.Lambda ( self.object_lambda.expand , input_shape = [ settings.state_size ] , name = 'Input_Lambda_1' ) )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        if RNN == 1 :

           self.model.add ( tf.keras.layers.SimpleRNN ( settings.hidden         ,
                                                        return_sequences = True , go_backwards = False             ,
                                                        name = 'RNN_relu_1'     , activation = self.settings.basic ))

           self.model.add ( tf.keras.layers.SimpleRNN ( settings.hidden          ,
                                                        return_sequences = False , go_backwards = False             ,
                                                        name = 'RNN_relu_2'      , activation = self.settings.basic ))
        else :

           if self.settings.conv_kernel  > 0 : self.model.add ( tf.keras.layers.Flatten ( name = 'Flatten' ) )

           if self.settings.conv_kernel  > 0 :
              self.model.add ( tf.keras.layers.Dense ( self.settings.hidden , activation = self.settings.basic , name = 'DenseDeep1' ) )

           else : self.model.add ( tf.keras.layers.Dense ( self.settings.hidden ,
                                   activation = self.settings.basic , name = 'DenseDeep1' , input_shape = [ settings.state_size ] ) )

           if self.settings.conv_kernel == 0 :

              self.model.add ( tf.keras.layers.Dense ( self.settings.hidden , activation = self.settings.basic , name = 'DenseDeep2' ) )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        self.model.add ( tf.keras.layers.Dense ( 1 , activation = None , name = 'Dense_Value' )) # kernel_initializer = self.ortog

        self.model.compile ( optimizer = self.optimizer , loss = 'mean_squared_error' , metrics = [ 'mse' , 'mae' , 'accuracy' ] )
        self.model.summary ()

    def setWeights     ( self , weights                             ) : self.model.set_weights ( weights )
    def training       ( self , states , labels , verbose = 0       ) :
        self.model.fit (        states , labels , verbose = verbose )

