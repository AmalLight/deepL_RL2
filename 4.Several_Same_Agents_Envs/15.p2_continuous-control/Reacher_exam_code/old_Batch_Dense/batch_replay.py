import tensorflow as tf
import numpy      as np
import threading

from collections import deque
import random

class ReplayBuffer () :

    def __init__     ( self , settings ) :
        self.settings       = settings
        self.maxlen         = settings.memory_deque

        self.memory   = deque ( maxlen = self.maxlen )
        self.memory_i = 0

    def add                ( self , states , actions , rewards , old_probs , advantages , values ) :
        self.memory.append (      ( states , actions , rewards , old_probs , advantages , values ) )

    def len_memory     ( self ) : return     len ( self.memory )
    def reset_memory_i ( self ) :                  self.memory_i = 0
    def shuffle        ( self ) : random.shuffle ( self.memory )
    def destroy        ( self ) : self.memory = deque ( maxlen = self.maxlen )

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def sample ( self ) :

        if self.len_memory () < self.settings.batch_size : return None

        batch_return = []
        for batch in range ( self.settings.batch_size ) :

            batch_return  += [[ tf.convert_to_tensor ( self.memory [ self.memory_i ][ 0 ] , dtype = float ) ,
                                tf.convert_to_tensor ( self.memory [ self.memory_i ][ 1 ] , dtype = float ) ,
                                tf.convert_to_tensor ( self.memory [ self.memory_i ][ 2 ] , dtype = float ) ,
                                tf.convert_to_tensor ( self.memory [ self.memory_i ][ 3 ] , dtype = float ) ,
                                tf.convert_to_tensor ( self.memory [ self.memory_i ][ 4 ] , dtype = float ) ]]
            self.memory_i += 1

        return batch_return
