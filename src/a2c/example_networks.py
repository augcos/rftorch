import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ExampleActor is an example of critic network for the DDPG agent
class ExampleActor(nn.Module):
    def __init__(self, lr=5e-5, input_dims=[4], fc1_dims=1024, fc2_dims=512, n_actions=2, 
                    chkpt_path="actor.pt"):
        super(ExampleActor, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_path = chkpt_path

        # full network block
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = F.softmax(self.pi(x))
        
        return pi
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))


# ExampleCritic is an example of critic network for the DDPG agent
class ExampleCritic(nn.Module):
    def __init__(self, lr=5e-5, input_dims=[4], fc1_dims=1024, fc2_dims=512, 
                    chkpt_path="critic.pt"):
        super(ExampleCritic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_path = chkpt_path
        
        # full network block
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)

        return v
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))