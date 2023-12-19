####################################
## Packages
####################################
import sys
import os
import math
import typing
import numpy as np
import torch
import torch.nn as nn

####################################
## Modules
####################################
sys.path.insert(0, os.path.abspath('.'))
from attention import Attention
from neural_net import NeuralNet

####################################
## Custom Datatype for Type Hints
####################################
NoneFloat = typing.Union[None, float]

####################################
## Encoding Module
####################################
class Encoding(nn.Module):

    def __init__(self,
    			dim: int,
    			num_heads: int=1,
    			hidden_chan_mul: float=4.,
    			qkv_bias: bool=False,
    			qk_scale: NoneFloat=None,
    			act_layer=nn.GELU, 
    			norm_layer=nn.LayerNorm):
        
        """ Encoding Block

            Args:
                dim (int): size of a single token
                num_heads(int): number of attention heads in MSA
                hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component
                qkv_bias (bool): determines if the qkv layer learns an addative bias
                qk_scale (NoneFloat): value to scale the queries and keys by; 
                                    if None, queries and keys are scaled by ``head_dim ** -0.5``
                act_layer(nn.modules.activation): torch neural network layer class to use as activation
                norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
        """

        super().__init__()

        ## Define Layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
	                            chan=dim,
	                            num_heads=num_heads,
	                            qkv_bias=qkv_bias,
	                            qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.neuralnet = NeuralNet(in_chan=dim,
                                hidden_chan=int(dim*hidden_chan_mul),
                                out_chan=dim,
                                act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.neuralnet(self.norm2(x))
        return x