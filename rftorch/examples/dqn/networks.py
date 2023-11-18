import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# DeepQNetwork is an example of Deep Q-Network for the DQN agent
class DeepQNetwork(nn.Module):
    def __init__(self, 
                 input_shape, 
                 n_actions, 
                 lr=0.00005, 
                 fc1_dims=128, 
                 fc2_dims=128):
        super(DeepQNetwork, self).__init__()
        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # block of dense layers
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # q_values are computed
        q_values = F.relu(self.fc1(state))
        q_values = F.relu(self.fc2(q_values))
        q_values = self.fc3(q_values)

        return q_values







# DuelingDeepQNetwork is an example of deep q-network with dueling architecture
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, 
                 input_shape, 
                 n_actions, lr=0.001, 
                 fc1_dims=128, 
                 fc2_dims=128):
        super(DuelingDeepQNetwork, self).__init__()
        self.input_shape = input_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        # block of dense layers
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.V = nn.Linear(self.fc2_dims, 1)
        self.A = nn.Linear(self.fc2_dims, self.n_actions)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # state and action values are computed
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.V(x)
        actions = self.A(x)

        return value, actions





# DuelingDeepQNetwork is an example of deep q-network with dueling architecture
class ConvDeepQNetwork(nn.Module):
    def __init__(self, 
                 input_shape,
                 n_actions, 
                 lr=0.001, 
                 conv1_dims=(16,3),
                 fc1_dims=128, 
                 fc2_dims=64):
        super(ConvDeepQNetwork, self).__init__()

        # block of dense layers
        self.conv1 = nn.Conv2d(input_shape[0], *conv1_dims)

        out = self.conv1(T.randn(input_shape[0], input_shape[1], input_shape[2]))

        self.fc1 = nn.Linear(T.numel(out), fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # state and action values are computed
        x = self.conv1(state)
        
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.V(x)
        actions = self.A(x)

        return value, actions
