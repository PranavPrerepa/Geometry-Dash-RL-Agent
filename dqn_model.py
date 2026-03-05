import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNAgent(nn.Module):
    def __init__(self, action_size=3):
        """
        Deep Q-Network for Geometry Dash.
        
        Args:
            action_size (int): Number of possible actions.
                             For GD, typically: 0 = Do nothing, 1 = Jump/Hold, (2 = Let go - depending on setup)
        """
        super(DQNAgent, self).__init__()
        
        # Input shape: (Batch Size, Channels (1 for grayscale), Height (128), Width (128))
        
        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully Connected Block
        # After conv layers, our 128x128 image becomes:
        # Conv1: ((128 - 8)/4)+1 = 31 -> 32x31x31
        # Conv2: ((31 - 4)/2)+1 = 14 -> 64x14x14
        # Conv3: ((14 - 3)/1)+1 = 12 -> 64x12x12
        # Flattened size: 64 * 12 * 12 = 9216
        
        self.fc1 = nn.Linear(9216, 512)
        
        # Output layer for Q-values of each action
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        """
        Calculates Q-values given a state/image.
        
        Args:
            x (torch.Tensor): Tensor of shape (B, 1, 128, 128) representing grayscale frames.
                              Should be normalized between 0.0 and 1.0.
        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Pass through Conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor before fully connected layers
        x = x.view(x.size(0), -1) 
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

# Example usage/test if run directly
if __name__ == "__main__":
    # Create the model expecting 2 actions: [Do Nothing, Jump]
    model = DQNAgent(action_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model created and moved to: {device}")
    
    # Create a dummy batch of 1 grayscale screen capture (1 batch, 1 channel, 128h, 128w)
    # This emulates the output from our screen_capture.py after normalization
    dummy_input = torch.rand((1, 1, 128, 128)).to(device)
    
    # Pass through model
    q_vals = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output Q-values: {q_vals}")
    print(f"Output shape (Batch Size, Actions): {q_vals.shape}")
