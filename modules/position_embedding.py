# POSITION EMBEDDING FOR VISION TRANSFORMERS
"""
Module definition for position embedding as decribed in Attention Is All You Need by Vaswani et al (2017)

Code is based on the publically available code for Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet by Yuan et al (2021)

Code here is identical to the code in Position Embeddings for Vision Transformers, Explained
"""

####################################
## Packages
####################################
import math
import typing
import numpy as np
import torch

####################################
## Position Embedding
####################################
def get_sinusoid_encoding(num_tokens, token_len):
	""" Make Sinusoid Encoding Table

		Args:
			num_tokens (int): number of tokens
			token_len (int): length of a token
			
		Returns:
			(torch.FloatTensor) sinusoidal position encoding table
	"""

	def get_position_angle_vec(i):
		return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

	sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)