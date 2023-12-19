####################################
## Packages
####################################
import typing
import torch
import torch.nn as nn

####################################
## Custom Datatype for Type Hints
####################################
NoneFloat = typing.Union[None, float]

####################################
## Neural Network Module
####################################
class NeuralNet(nn.Module):
    def __init__(self,
    			in_chan: int,
    			hidden_chan: NoneFloat=None,
    			out_chan: NoneFloat=None,
    			act_layer = nn.GELU):
        """ Neural Network Module

            Args:
                in_chan (int): number of channels (features) at input
                hidden_chan (NoneFloat): number of channels (features) in the hidden layer;
                                        if None, number of channels in hidden layer is the same as the number of input channels
                out_chan (NoneFloat): number of channels (features) at output;
                                        if None, number of output channels is same as the number of input channels
                act_layer(nn.modules.activation): torch neural network layer class to use as activation
        """

        super().__init__()

        ## Define Number of Channels
        hidden_chan = hidden_chan or in_chan
        out_chan = out_chan or in_chan

        ## Make Parameters Integers
        assert isinstance(hidden_chan, int) or hidden_chan.is_integer(), "Hidden channels in Neural Network module must be an integer"
        in_chan = int(in_chan)
        hidden_chan = int(hidden_chan)
        out_chan = int(out_chan)

        ## Define Layers
        self.fc1 = nn.Linear(in_chan, hidden_chan)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_chan, out_chan)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x