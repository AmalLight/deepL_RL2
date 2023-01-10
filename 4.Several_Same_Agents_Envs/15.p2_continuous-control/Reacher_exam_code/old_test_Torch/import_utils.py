import numpy as np
import torch.nn as nn


class Batcher:
    
    def __init__ ( self  , batch_size , data ):
        self.batch_size  = batch_size
        self.data        =       data
        self.num_entries = len ( data [ 0 ] )
        self.reset ()

    def reset ( self ) :
        self.batch_start = 0
        self.batch_end   = self.batch_start + self.batch_size

    def end ( self ) : return self.batch_start >= self.num_entries

    def next_batch ( self ) :
        batch = []

        for d in self.data : batch.append ( d [ self.batch_start : self.batch_end ] )

        self.batch_start                        = self.batch_end
        self.batch_end = min ( self.batch_start + self.batch_size , self.num_entries )
        return batch

    def shuffle ( self ) :

        indices = np.arange         ( self.num_entries )
        np.random.shuffle ( indices )
        self.data  =  [ d [ indices ] for d in self.data ]
