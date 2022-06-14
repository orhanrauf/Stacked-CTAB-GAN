
import numpy as np
import pandas as pd
from model.synthesizer.ctabgan_synthesizer import Condvec, random_choice_prob_index_sampling


class StackedCondvecFactory(Condvec):
    """
    This class is responsible for sampling conditional vectors to be supplied to the generator
    and kept throughout the StackGAN Layers.

    Variables:
    1) model -> list containing an index of highlighted categories in their corresponding one-hot-encoded represenations
    2) interval -> an array holding the respective one-hot-encoding starting positions and sizes     
    3) n_col -> total no. of one-hot-encoding representations
    4) n_opt -> total no. of distinct categories across all one-hot-encoding representations
    5) p_log_sampling -> list containing log of probability mass distribution of categories within their respective one-hot-encoding representations
    6) p_sampling -> list containing probability mass distribution of categories within their respective one-hot-encoding representations
    7) batch size -> size of batch

    Methods:
    1) __init__() -> takes transformed input data with respective column information to compute class variables with super method
    2) sample_train() -> used to sample the conditional vector during training of the modelwhich also saves condvec for layers n>1
    3) sample_next_layers() -> used to sample the previously generated conditional vector during training for layers n>1
    """
    
    def __init__(self, data, output_info, batch_size):
        
        super().__init__(data, output_info, batch_size)
        self.generated_condvecs = []
        
    def sample_train(self):
        condvec, mask, idx, opt1prime = super().sample_train()
        # save condvec and its metas
        if not self.generated_condvecs:
            self.generated_condvecs = [(condvec, mask, idx, opt1prime)]
        else:
            self.generated_condvecs.append((condvec,mask, idx, opt1prime)) 
        
        return condvec, mask, idx, opt1prime
    
    def sample(self):
        condvec = super().sample()
        if not self.generated_condvecs:
            self.generated_condvecs = [condvec]
        else:
            self.generated_condvecs.append(condvec) 
        
        return condvec
        
    def sample_next_layers(self, step):
        """
        Used to create the conditional vectors for feeding it to the generator during training
        
        Inputs:
        1) step -> step number as used in the batched training
        
        Outputs:
        1) vec -> a matrix containing a conditional vector for each data point to be generated 
        2) mask -> a matrix to identify chosen one-hot-encodings across the batch
        3) idx -> list of chosen one-hot encoding across the batch
        4) opt1prime -> selected categories within chosen one-hot-encodings

        """
        
        return self.generated_condvecs[step]
        
        
        
        
        
        
        