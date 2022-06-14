import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ExampleActor is an example of critic network for the DDPG agent
class ExampleActor(nn.Module):
    def __init__(self, lr=0.000025, input_dims=[8], fc1_dims=400, fc2_dims=300, n_actions=2, 
                    chkpt_path="actor.pt"):
        super(ExampleActor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_path = chkpt_path

        # first block: dense layer and batch normalization
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # seconds block: dense layer and batch normalization
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # third block: dense layer and batch normalization
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # continuous action value is computed
        action = self.fc1(state)
        action = self.bn1(action)
        action = F.relu(action)
        action = self.fc2(action)
        action = self.bn2(action)
        action = F.relu(action)
        action = T.tanh(self.mu(action))

        return action
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))


# ExampleCritic is an example of critic network for the DDPG agent
class ExampleCritic(nn.Module):
    def __init__(self, lr=0.00025, input_dims=[8], fc1_dims=400, fc2_dims=300, n_actions=2, 
                    chkpt_path="critic.pt"):
        super(ExampleCritic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.chkpt_path = chkpt_path
        
        # first block: dense layer and batch normalization
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # seconds block: dense layer and batch normalization
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # third block: dense layer and batch normalization
        self.q = nn.Linear(self.fc2_dims, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        # action block: dense layer and batch normalization
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # state is passed through the first and second blocks
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        # state is passed through the action block
        action_value = F.relu(self.action_value(action))
        
        # output of the second and action blocks is passed through the third block
        q_value = F.relu(T.add(state_value, action_value))
        q_value = self.q(q_value)

        return q_value
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))