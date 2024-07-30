import torch.nn as nn
import torch

class Net(nn.Module):
    
    def __init__(self, layers):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # input layer
        self.input_layer = nn.Linear(in_features=layers[0], out_features=layers[1], bias=True)

        # hidden layers
        for i in range(1, len(layers)-2):
            self.hidden_layers.append(nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=True))

        # output layer
        self.output_layer = nn.Linear(in_features=layers[-2], out_features=layers[-1], bias=False)

        # activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, x):

        # input layer
        x = self.act2(self.input_layer(x))

        # hidden layers
        for self.hidden_layer in self.hidden_layers:
            x = self.act2(self.hidden_layer(x))

        # output layer
        x = self.output_layer(x)

        return x


class ResNet(nn.Module):
    
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # input layer
        self.input_layer = nn.Linear(in_features=layers[0], out_features=layers[1], bias=True)

        # hidden layers
        for i in range(1, len(layers)-2):
            self.hidden_layers.append(nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=True))

        # output layer
        self.output_layer = nn.Linear(in_features=layers[-2], out_features=layers[-1], bias=False)

        # activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, x):

        # input layer
        identity = x

        out = self.act1(self.input_layer(x))

        # hidden layers
        for self.hidden_layer in self.hidden_layers:
            out = self.act2(self.hidden_layer(out))
            # If dimensions change, apply the downsample operation
            if self.downsample is not None:
                identity = self.downsample(x)
            out +=identity

        # output layer
        x = self.output_layer(x)

        return x

