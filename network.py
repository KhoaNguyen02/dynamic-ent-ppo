import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=False, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.discrete = discrete

        # Define the actor (policy) network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        if not discrete:
            self.std = nn.Parameter(0.001 * torch.ones(action_dim))

    def forward(self, state):
        """Forward pass of the network

        Parameters
        ----------
        state : torch.Tensor
            The input state tensor

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        x = self.network(state)
        if self.discrete:
            return F.softmax(x, dim=-1)
        else:
            return x

    def get_action(self, state):
        """Sample an action from the policy

        Parameters
        ----------
        state : torch.Tensor
            Current state of the environment

        Returns
        -------
        torch.Tensor
            The sampled action.
            
        """
        if self.discrete:
            probs = self.forward(state)
            dist = Categorical(probs)
            action = dist.sample()
        else:
            mean = self.forward(state)
            std = torch.exp(self.std).expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample()
        return action

    def log_probs(self, state, action):
        """Calculate the log probabilities of the given policy (state, action)

        Parameters
        ----------
        state : torch.Tensor
            Current state of the environment
        action : torch.Tensor
            The action taken in the environment

        Returns
        -------
        torch.Tensor
            The log probabilities of the given policy (state, action)
        """
        if self.discrete:
            probs = self.forward(state)
            dist = Categorical(probs)
            return dist.log_prob(action)
        else:
            mean = self.forward(state)
            std = torch.exp(self.std).expand_as(mean)
            dist = Normal(mean, std)
            return dist.log_prob(action).sum(dim=-1)
        
    def entropy(self, state):
        """Calculate the entropy of the policy

        Parameters
        ----------
        state : torch.Tensor
            Current state of the environment

        Returns
        -------
        torch.Tensor
            The resulting entropy
        """
        if self.discrete:
            probs = self.forward(state)
            dist = Categorical(probs)
            return dist.entropy().mean()
        else:
            mean = self.forward(state)
            std = torch.exp(self.std).expand_as(mean)
            dist = Normal(mean, std)
            return dist.entropy().mean()

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()

        # Define the critic (value) network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """Forward pass of the network

        Parameters
        ----------
        state : torch.Tensor
            The input state tensor

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        return self.network(state).squeeze(-1)