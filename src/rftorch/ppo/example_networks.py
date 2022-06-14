import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# ExampleActor is an example of critic network for the DDPG agent
class ExampleActor(nn.Module):
    def __init__(self, lr=0.0001, input_dims=[4], fc1_dims=256, fc2_dims=256, n_actions=2, 
                    chkpt_path="actor.pt"):
        super(ExampleActor, self).__init__()
        self.chkpt_path = chkpt_path

        # full network block
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))


# ExampleCritic is an example of critic network for the DDPG agent
class ExampleCritic(nn.Module):
    def __init__(self, lr=0.0001, input_dims=[4], fc1_dims=256, fc2_dims=256, 
                    chkpt_path="critic.pt"):
        super(ExampleCritic, self).__init__()
        self.chkpt_path = chkpt_path
        
        # full network block
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        # ADAM optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # device is set and network is moved to device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
        
    # save_checkpoint saves the model
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_path)

    # load_checkpoint loads the model
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_path))