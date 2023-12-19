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
import timm.layers

####################################
## Modules
####################################
sys.path.insert(0, os.path.abspath('.'))
from tokens_to_token import Tokens2Token
from ViT import ViT_Backbone

####################################
## T2T-ViT Model
####################################
class T2T_ViT(nn.Module):
    def __init__(self, 
                img_size: tuple[int, int, int]=(1, 1700, 500),
                softsplit_kernels: tuple[int, int, int]=(31, 3, 3),
                preds: int=1,
                token_len: int=768,
                token_chan:  int=64,
                num_heads: int=1,
                T2T_hidden_chan_mul: float=1.,
                Encoding_hidden_chan_mul: float=4.,
                depth: int=12,
                qkv_bias=False,
                qk_scale=None,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):

        """ Tokens-to-Token VisTransformer Model

            Args:
                img_size (tuple[int, int, int]): size of input (channels, height, width)
                softsplit_kernels (tuple[int int, int]): size of the square kernel for each of the soft split layers, sequentially
                preds (int): number of predictions to output
                token_len (int): desired length of an output token
                token_chan (int): number of token channels inside the TokenTransformers
                num_heads(int): number of attention heads in MSA (only works if =1)
                T2T_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Tokens-to-Token (T2T) Module
                Encoding_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Encoding Module
                depth (int): number of encoding blocks in the model
                qkv_bias (bool): determines if the qkv layer learns an addative bias
                qk_scale (NoneFloat): value to scale the queries and keys by; 
                                    if None, queries and keys are scaled by ``head_dim ** -0.5``
                act_layer(nn.modules.activation): torch neural network layer class to use as activation
                norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
        """

        super().__init__()

        ## Defining Parameters
        self.img_size = img_size
        C, H, W = self.img_size
        self.softsplit_kernels = softsplit_kernels
        self.token_len = token_len
        self.token_chan = token_chan
        self.num_heads = num_heads
        self.T2T_hidden_chan_mul = T2T_hidden_chan_mul
        self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
        self.depth = depth

        ## Defining Tokens-to-Token Module
        self.tokens_to_token = Tokens2Token(img_size = self.img_size, 
                                            softsplit_kernels = self.softsplit_kernels,
                                            num_heads = self.num_heads,
							    			token_chan = self.token_chan,
							    			token_len = self.token_len,
							    			hidden_chan_mul = self.T2T_hidden_chan_mul,
							    			qkv_bias = qkv_bias,
							    			qk_scale = qk_scale,
							    			act_layer = act_layer,
							    			norm_layer = norm_layer)
        self.num_tokens = self.tokens_to_token.num_tokens

        ## Defining Token Processing Components
        self.vit_backbone = ViT_Backbone(preds = preds,
										token_len = self.token_len,
										num_heads = self.num_heads,
										Encoding_hidden_chan_mul = self.Encoding_hidden_chan_mul,
										depth = self.depth,
										qkv_bias = qkv_bias,
						    			qk_scale = qk_scale,
						    			act_layer = act_layer,
						    			norm_layer = norm_layer)
        

        ## Initialize the Weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Initialize the weights of the linear layers & the layernorms
		"""
        ## For Linear Layers
        if isinstance(m, nn.Linear):
            ## Weights are initialized from a truncated normal distrobution
            timmm.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                ## If bias is present, bias is initialized at zero
                nn.init.constant_(m.bias, 0)
        ## For Layernorm Layers
        elif isinstance(m, nn.LayerNorm):
            ## Weights are initialized at one
            nn.init.constant_(m.weight, 1.0)
            ## Bias is initialized at zero
            nn.init.constant_(m.bias, 0)
            
    @torch.jit.ignore ##Tell pytorch to not compile as TorchScript
    def no_weight_decay(self):
        """ Used in Optimizer to ignore weight decay in the class token
        """
        return {'cls_token'}

    def forward(self, x):
        x = self.tokens_to_token(x)
        x = self.vit_backbone(x)
        return x