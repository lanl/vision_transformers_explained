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
from encoding import Encoding

####################################
## ViT Patch Embedding
####################################
class Patch_Embedding(nn.Module):
	def __init__(self,
				img_size: tuple[int, int, int]=(1, 400, 100),
				patch_size: int=50,
				token_len: int=768):

		""" Patch Embedding Module

			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				patch_size (int): the side length of a square patch
				token_len (int): desired length of an output token
		"""

		## Defining Parameters
		self.img_size = img_size
		C, H, W = self.img_size
		self.patch_size = patch_size
		self.token_len = token_len
		assert H % self.patch_size == 0, 'Height of image must be evenly divisible by patch size.'
		assert W % self.patch_size == 0, 'Width of image must be evenly divisible by patch size.'
		self.num_tokens = (H / self.patch_size) * (W / self.patch_size)

		## Defining Layers
		self.split = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=0)
		self.project = nn.Linear(self.patch_size**2, token_len)

	def forward(self, x):
		x = self.split(x)
		x = self.project(x)
		return x


####################################
## ViT Backbone Module
####################################
class ViT_Backbone(nn.Module):
	def __init__(self,
				preds: int=1,
				token_len: int=768,
				num_heads: int=1,
				Encoding_hidden_chan_mul: float=4.,
				depth: int=12,
				qkv_bias=False,
				qk_scale=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" VisTransformer Backbone

			Args:
				preds (int): number of predictions to output
				token_len (int): length of a token
				num_heads(int): number of attention heads in MSA
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
		self.num_heads = num_heads
		self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
		self.depth = depth

		## Defining Token Processing Components
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_len))
		self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(num_tokens=self.num_tokens+1, token_len=self.token_len), requires_grad=False)

		## Defining Encoding blocks
		self.blocks = nn.ModuleList([Encoding(dim = self.token_len, 
											num_heads = self.num_heads,
											hidden_chan_mul = self.Encoding_hidden_chan_mul,
											qkv_bias = qkv_bias,
											qk_scale = qk_scale,
											act_layer = act_layer,
											norm_layer = norm_layer)
									for i in range(self.depth)])

		## Defining Prediction Processing
		self.norm = norm_layer(self.token_len)
		self.head = nn.Linear(self.token_len, preds)

		## Make the class token sampled from a truncated normal distrobution 
		timm.layers.trunc_normal_(self.cls_token, std=.02)

	def forward(self, x):
		## Assumes x is already tokenized

		## Get Batch Size
		B = x.shape[0]
		## Concatenate Class Token
		x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
		## Add Positional Embedding
		x = x + self.pos_embed
		## Run Through Encoding Blocks
		for blk in self.blocks:
			x = blk(x)
		## Take Norm
		x = self.norm(x)
		## Make Prediction on Class Token
		x = self.head(x[:, 0])
		return x

####################################
## ViT Model
####################################
class ViT_Model(nn.Module):
	def __init__(self,
				img_size: tuple[int, int, int]=(1, 400, 100),
				patch_size: int=50,
				token_len: int=768,
				preds: int=1,
				num_heads: int=1,
				Encoding_hidden_chan_mul: float=4.,
				depth: int=12,
				qkv_bias=False,
				qk_scale=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" VisTransformer Model

			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				patch_size (int): the side length of a square patch
				token_len (int): desired length of an output token
				preds (int): number of predictions to output
				num_heads(int): number of attention heads in MSA
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
		self.patch_size = patch_size
		self.token_len = token_len
		self.num_heads = num_heads
		self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
		self.depth = depth

		## Defining Patch Embedding Module
		self.patch_embed = Patch_Embedding(img_size,
											patch_size,
											token_len)

		## Defining ViT Backbone
		self.backbone = ViT_Backbone(preds,
									self.token_len,
									self.num_heads,
									self.Encoding_hidden_chan_mul,
									self.depth,
									qkv_bias,
									qk_scale,
									act_layer,
									norm_layer)
		## Initialize the Weights
		self.apply(self._init_weights)

	def _init_weights(self, m):
		""" Initialize the weights of the linear layers & the layernorms
		"""
		## For Linear Layers
		if isinstance(m, nn.Linear):
			## Weights are initialized from a truncated normal distrobution
			timm.layers.trunc_normal_(m.weight, std=.02)
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
		x = self.patch_embed(x)
		x = self.backbone(x)
		return x