import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim) # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, action_dim) # Output layer

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # Define the model
    net = DQN(12, 2)
    state = torch.randn(10, 12)
    output = net(state)
    print(output)
        