#TOKENS-TO-TOKEN VISION TRANSFORMER
"""
Module definition for a Tokens-to-Token Vision Transformer (T2T-ViT) as decribed in Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet by Yuan et al (2021)

Code is based on the publically available code for Tokens-to-Token ViT

Code here is identical to the code in Tokens-to-Token Vision Transformers, Explained
"""

####################################
## Packages
####################################
import math
import typing
import numpy as np
import torch
import torch.nn as nn
import timm.layers as timmm 

from attention import Attention
from position_embedding import get_sinusoid_encoding
import vit

####################################
## Custom Datatype for Type Hints
####################################
NoneFloat = typing.Union[None, float]

####################################
## Token Transformer
####################################
class TokenTransformer(nn.Module):

	def __init__(self,
				dim: int,
				chan: int,
				num_heads: int,
				hidden_chan_mul: float=1.,
				qkv_bias: bool=False,
				qk_scale: NoneFloat=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" Token Transformer Module

			Args:
				dim (int): size of a single token
				chan (int): resulting size of a single token 
				num_heads (int): number of attention heads in MSA 
				hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet module
				qkv_bias (bool): determines if the attention qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
				if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation in the NeuralNet module
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Define Layers
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim,
		chan=chan,
		num_heads=num_heads,
		qkv_bias=qkv_bias,
		qk_scale=qk_scale)
		self.norm2 = norm_layer(chan)
		self.neuralnet = vit.NeuralNet(in_chan=chan,
		hidden_chan=int(chan*hidden_chan_mul),
		out_chan=chan,
		act_layer=act_layer)

	def forward(self, x):
		x = self.attn(self.norm1(x))
		x = x + self.neuralnet(self.norm2(x))
		return x

####################################
## Tokens-to-Token (T2T) Module
####################################
class Tokens2Token(nn.Module):
	def __init__(self, 
				img_size: tuple[int, int, int]=(1, 1000, 300), 
				token_chan:  int=64,
				token_len: int=768,):
	
		""" Tokens-to-Token Module
			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				token_chan (int): number of token channels inside the TokenTransformers
				token_len (int): desired length of an output token
		"""

		super().__init__()

		## Seperating Image Size
		C, H, W = img_size
		self.token_chan = token_chan
		## Dimensions: (channels, height, width)

		## Define the Soft Split Layers
		self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
		self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

		## Determining Number of Output Tokens
		W, H, _ = count_tokens(w=W, h=H, k=7, s=4, p=2)
		W, H, _ = count_tokens(w=W, h=H, k=3, s=2, p=1)
		_, _, T = count_tokens(w=W, h=H, k=3, s=2, p=1)
		self.num_tokens = T


		## Define the Transformer Layers
		self.transformer1 = TokenTransformer(dim= C * 7 * 7, 
							chan=token_chan,
							num_heads=1,
							hidden_chan_mul=1.0)
		self.transformer2 = TokenTransformer(dim=token_chan * 3 * 3, 
							chan=token_chan, 
							num_heads=1, 
							hidden_chan_mul=1.0)

		## Define the Projection Layer
		self.project = nn.Linear(token_chan * 3 * 3, token_len)
  
	def forward(self, x):

		B, C, H, W = x.shape
		## Dimensions: (batch, channels, height, width)

		## Initial Soft Split
		x = self.soft_split0(x).transpose(1, 2)

		## Token Transformer 1
		x = self.transformer1(x)

		## Reconstruct 2D Image
		W, H, _ = count_tokens(w=W, h=H, k=7, s=4, p=2)
		x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

		## Soft Split 1
		x = self.soft_split1(x).transpose(1, 2)

		## Token Transformer 2
		x = self.transformer2(x)

		## Reconstruct 2D Image
		W, H, _ = count_tokens(w=W, h=H, k=3, s=2, p=1)
		x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

		## Soft Split 2
		x = self.soft_split2(x).transpose(1, 2)

		## Project Tokens to desired length
		x = self.project(x)

		return x

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
		self.vit_backbone = vit.ViT_Backbone(preds = preds,
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