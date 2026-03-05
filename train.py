import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

from env import GeometryDashEnv
from replay_buffer import ReplayBuffer
from dqn_model import DQNAgent

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99             # Discount factor for future rewards
EPSILON_START = 1.0      # Initial exploration rate (100% random)
EPSILON_END = 0.05       # Final exploration rate (5% random)
EPSILON_DECAY = 10000    # How many steps to decay over
LEARNING_RATE = 1e-4
TARGET_UPDATE = 50       # How often to update the target network
MEMORY_CAPACITY = 20000

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Initialize Modules
env = GeometryDashEnv()
# Create two identical models - Policy (the one training) and Target (provides stable Q-values for the Bellman eqn)
policy_net = DQNAgent(action_size=env.action_space_size).to(device)
target_net = DQNAgent(action_size=env.action_space_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

memory = ReplayBuffer(capacity=MEMORY_CAPACITY)

def select_action(state, steps_done):
    """Epsilon Greedy action selection. Sometimes random, sometimes driven by the model."""
    # Calculate current epsilon
    # The decay slows down exponentially over time
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
              np.exp(-1. * steps_done / EPSILON_DECAY)
              
    if random.random() > epsilon:
        # Exploit: Use the Model
        with torch.no_grad():
            # State is (1, 128, 128). Add Batch dimension -> (1, 1, 128, 128)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            # Find the action with the highest Q-value
            return q_values.max(1)[1].item()
    else:
        # Explore: Random Action
        return random.randint(0, env.action_space_size - 1)

def optimize_model():
    """Takes a batch from memory and updates the neural network weights."""
    if len(memory) < BATCH_SIZE:
        return 0.0 # Not enough memory to train yet
        
    # Get random batch of experiences
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(BATCH_SIZE, device)
    
    # Normalize rewards to prevent exploding Q-values
    # (e.g. survival = 0.01, death = -1.0)
    reward_batch = torch.clamp(reward_batch / 100.0, -1.0, 1.0)
    
    # 1. Compute Q(s_t, a) - the model's prediction for the action we ACTUALLY took
    # policy_net(state_batch) outputs Q-values for ALL actions. 
    # gather() picks just the Q-value for the specific action we took at that moment.
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # 2. Compute max Q(s_{t+1}, a) from the stable TARGET network
    # We do a max(1)[0] to find the highest Q-value the target network predicts for the next state
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # If the episode ended (done is 1.0), there is no future reward, so zero it out
    next_state_values = next_state_values * (1 - done_batch)
    
    # 3. Compute expected Q values (The Bellman Equation)
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)
    
    # 4. Compute Loss using Huber Loss (SmoothL1) instead of raw MSE to prevent exploding gradients
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # 5. Optimize
    optimizer.zero_grad()
    loss.backward()
    
    # Strict Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            
    optimizer.step()
    
    return loss.item()

def train(num_episodes=5000):
    steps_done = 0
    
    print("\n--- Starting Training Loop ---")
    print("Ensure Geometry Dash is open and visible on screen.")
    print("Press Ctrl+C to stop training safely.\n")
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            # Let the episode run as long as the AI is alive
            import itertools
            import cv2  # Just for displaying what the AI sees
            
            for t in itertools.count():
                
                # --- SHOW WHAT THE AI SEES ---
                # State is (1, 128, 128) float [0.0, 1.0]. 
                # Squeeze to (128, 128) and scale back up to [0, 255] for OpenCV to display
                vision_array = (state[0] * 255).astype(np.uint8)
                cv2.imshow("AI Vision (128x128)", vision_array)
                # Keep the window refreshing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("q pressed. Stopping training safely...")
                    raise KeyboardInterrupt
                # ------------------------------
                
                # Pick and execute action
                action = select_action(state, steps_done)
                next_state, reward, done = env.step(action)
                total_reward += reward
                steps_done += 1
                
                # Keep the state intact so it maintains the (1, 128, 128) shape in the replay buffer
                memory.push(state, action, reward, next_state, float(done))
                
                # Move to next state
                state = next_state
                
                # Perform model optimization
                loss = optimize_model()
                
                # Periodically update the target network
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
                # Log the step
                action_name = "Jump/Hold" if action == 1 else "Nothing  "
                print(f"Ep {episode+1:3} | Step {t:4} | Action: {action_name} | Reward: {reward:6.1f} | Loss: {loss:.4f}")
                    
                if done:
                    print(f"Episode {episode+1} | Steps: {t} | Reward: {total_reward} | Loss: {loss:.4f} | Epsilon: {max(EPSILON_END, EPSILON_START * np.exp(-1. * steps_done / EPSILON_DECAY)):.3f}")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        
    finally:
        # Save model
        torch.save(policy_net.state_dict(), "gd_dqn_model.pth")
        print("Model saved to gd_dqn_model.pth")
        
if __name__ == "__main__":
    # Add a short delay to let user tab into the game
    print("Starting in 3 seconds... Tab to Geometry Dash!")
    time.sleep(3)
    train()
