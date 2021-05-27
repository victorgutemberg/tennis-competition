import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
        super().__init__()

        # input layer
        self.input_layer = nn.Linear(state_size, hidden_layers[0])

        # hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:])])

        # output layer
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
        self.output_activation_fn = F.tanh

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_activation_fn(self.output_layer(x))
    
    def load_checkpoint(self):
        checkpoint_path = os.path.join('checkpoints', 'model_actor.pth')
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[256, 128]):
        super().__init__()

        # input layer
        self.input_layer = nn.Linear(state_size, hidden_layers[0])

        # hidden layers
        self.hidden_layers = nn.ModuleList()
        add_action_size = True
        for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            # the action is not included until the second hidden layer
            if add_action_size:
                h1 += action_size
                add_action_size = False
            self.hidden_layers.append(nn.Linear(h1, h2))

        # output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, state, action):
        x = F.relu(self.input_layer(state))
        add_action_size = True
        for hidden_layer in self.hidden_layers:
            # the action is not included until the second hidden layer
            if add_action_size:
                x = torch.cat((x, action), dim=1)
                add_action_size = False
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)

    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
