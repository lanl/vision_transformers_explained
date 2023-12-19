####################################
## Packages
####################################
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
## Count Tokens
####################################
def count_tokens(w, h, k, s, p):
    """ Function to count how many tokens are produced from a given soft split

        Args:
            w (int): starting width
            h (int): starting height
            k (int): kernel size
            s (int): stride size
            p (int): padding size

        Returns:
            new_w (int): number of tokens along the width
            new_h (int): number of tokens along the height
            total (int): total number of tokens created

        See Also: 
        Formula taken from 
        https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
        Assuming a 2D input, dilation = 1, and symmetric padding, kernel, and stride
    """

    new_w = int(math.floor(((w + 2*p - (k-1) -1)/s)+1))
    new_h = int(math.floor(((h + 2*p - (k-1) -1)/s)+1))
    total = new_w * new_h

    return new_w, new_h, total

####################################
## Token Transformer Module
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
                dim (int):input size of a single token
                chan (int): resulting size of a single token
                num_heads (int): number of attention heads in MSA
                hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet module
                qkv_bias (bool): determines if the QKV layer in the Attention Module learns an addative bias
                qk_scale (NoneFloat): value to scale the Attention module queries and keys by; 
                                    if None, queries and keys are scaled by ``head_dim ** -0.5``
                act_layer(nn.modules.activation): torch neural network layer class to use as activation in the NeuralNet module
                norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
        """

        super().__init__()

        ## Define Layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
        						chan=chan,
        						num_heads=num_heads,
        						qkv_bias=qkv_bias,
        						qk_scale=qk_scale)
        self.norm2 = norm_layer(chan)
        self.neuralnet = NeuralNet(in_chan=chan,
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
    			img_size: tuple[int, int, int]=(1, 1700, 500), 
                softsplit_kernels: tuple[int, int, int]=(31, 3, 3),
    			num_heads: int=1,
    			token_chan:  int=64,
    			token_len: int=768,
    			hidden_chan_mul: float=1.,
    			qkv_bias: bool=False,
    			qk_scale: NoneFloat=None,
    			act_layer=nn.GELU,
    			norm_layer=nn.LayerNorm):

        """ Tokens-to-Token Module

			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
                softsplit_kernels (tuple[int int, int]): size of the square kernel for each of the soft split layers, sequentially
				num_heads (int): number of attention heads in MSA
				token_chan (int): number of token channels inside the TokenTransformers
				token_len (int): desired length of an output token
				hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet module
                qkv_bias (bool): determines if the QKV layer in the Attention Module learns an addative bias
                qk_scale (NoneFloat): value to scale the Attention module queries and keys by; 
                                    if None, queries and keys are scaled by ``head_dim ** -0.5``
                act_layer(nn.modules.activation): torch neural network layer class to use as activation in the NeuralNet module
                norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

        super().__init__()

        ## Seperating Image Size
        C, H, W = img_size
        self.token_chan = token_chan
        ## Dimensions: (channels, height, width)

        ## Define the Soft Split Layers
        self.k0, self.k1, self.k2 = softsplit_kernels
        self.s0, self.s1, self.s2 = [int((k+1)/2) for k in softsplit_kernels]
        self.p0, self.p1, self.p2 = [int((k+1)/4) for k in softsplit_kernels]
        self.soft_split0 = nn.Unfold(kernel_size=(self.k0, self.k0), stride=(self.s0, self.s0), padding=(self.p0, self.p0))
        self.soft_split1 = nn.Unfold(kernel_size=(self.k1, self.k1), stride=(self.s1, self.s1), padding=(self.p1, self.p1))
        self.soft_split2 = nn.Unfold(kernel_size=(self.k2, self.k2), stride=(self.s2, self.s2), padding=(self.p2, self.p2))

        ## Determining Number of Output Tokens
        W, H, _ = count_tokens(w=W, h=H, k=self.k0, s=self.s0, p=self.p0)
        W, H, _ = count_tokens(w=W, h=H, k=self.k1, s=self.s1, p=self.p1)
        _, _, T = count_tokens(w=W, h=H, k=self.k2, s=self.s2, p=self.p2)
        self.num_tokens = T


        ## Define the Transformer Layers
        self.transformer1 = TokenTransformer(dim =  C * self.k0 * self.k0,
							    			chan = token_chan,
							    			num_heads = num_heads,
							    			hidden_chan_mul = hidden_chan_mul,
							    			qkv_bias = qkv_bias,
							    			qk_scale = qk_scale,
							    			act_layer = act_layer,
							    			norm_layer = norm_layer)

        self.transformer2 = TokenTransformer(dim =  token_chan * self.k1 * self.k1,
							    			chan = token_chan,
							    			num_heads = num_heads,
							    			hidden_chan_mul = hidden_chan_mul,
							    			qkv_bias = qkv_bias,
							    			qk_scale = qk_scale,
							    			act_layer = act_layer,
							    			norm_layer = norm_layer)

        ## Define the Projection Layer
        self.project = nn.Linear(token_chan * self.k2 * self.k2, token_len)

    def forward(self, x):

        B, C, H, W = x.shape
        ### Dimensions: (batch, channels, height, width)

        ## Initial Soft Split
        x = self.soft_split0(x).transpose(1, 2)

        ## Token Transformer 1
        x = self.transformer1(x)

        ## Reconstruct 2D Image
        W, H, _ = count_tokens(w=W, h=H, k=self.k0, s=self.s0, p=self.p0)
        x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

        ## Soft Split 1
        x = self.soft_split1(x).transpose(1, 2)

        ## Token Transformer 2
        x = self.transformer2(x)

        ## Reconstruct 2D Image
        W, H, _ = count_tokens(w=W, h=H, k=self.k1, s=self.s1, p=self.p1)
        x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

        ## Soft Split 2
        x = self.soft_split2(x).transpose(1, 2)

        ## Project Tokens to desired length
        x = self.project(x)

        return x