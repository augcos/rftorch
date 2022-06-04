import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ExampleDeepQNetwork is an example of deep q-network for the DQN agent
class ExampleDeepQNetwork(nn.Module):
    def __init__(self, lr=0.0005, input_dims=[8], fc1_dims=256, fc2_dims=256, n_actions=4,
                chkpt_path="dqn.pt"):
        super(ExampleDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_path = chkpt_path

        # block of dense layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
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

    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))


# ExampleDeepQNetwork is an example of deep q-network for the DQN agent
class ExampleDuelingDeepQNetwork(nn.Module):
    def __init__(self, lr=0.001, input_dims=[8], fc1_dims=256, fc2_dims=256, n_actions=4,
                chkpt_path="dqn.pt"):
        super(ExampleDuelingDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_path = chkpt_path

        # block of dense layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.V = nn.Linear(self.fc2_dims, 1)
        self.A = nn.Linear(self.fc2_dims, self.n_actions)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # value and action values are computed
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.V(x)
        actions = self.A(x)

        return value, actions

    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))