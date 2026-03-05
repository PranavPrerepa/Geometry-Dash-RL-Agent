# Geometry Dash RL AI 🤖🧊

A PyTorch-based Reinforcement Learning intelligent agent that learns to play **Geometry Dash**. This project uses a Deep Q-Network (DQN) architecture to process raw screen pixels and compute optimal jump timings to beat levels without accessing the game's internal memory.

## 🚀 Key Features
- **Computer Vision Pipeline**: High-speed, 60 FPS screen capturing using `mss` and OpenCV.
- **Deep Q-Network (DQN)**: A Convolutional Neural Network (CNN) implemented in PyTorch that acts as a feature extractor to learn from 128x128 grayscale frames.
- **Experience Replay Buffer**: Improves sample efficiency and stabilizes training by storing past state transitions.
- **Custom RL Environment**: An OpenAI Gym-like wrapper that automates interactions using `pyautogui` and detects deaths via template matching and frame differences.
- **Epsilon-Greedy Exploration**: Balances exploring new strategies vs. exploiting the current best known policy.

## 📁 Project Structure

```text
GD_RL/
├── env.py                  # Custom RL environment wrapper to interface with the game.
├── dqn_model.py            # PyTorch architecture for the CNN DQN agent.
├── screen_capture.py       # High FPS background window capturing script.
├── replay_buffer.py        # Experience replay buffer logic.
├── train.py                # Main training loop with Bellman Equation implementation.
├── capture_death_screen.py # Script to capture 'Attempt' text for death detection.
└── requirements.txt        # Project dependencies.
```

## 🛠️ Prerequisites
- Python 3.8+
- [Geometry Dash](https://store.steampowered.com/app/322170/Geometry_Dash/) (Must be visible on screen during training)
- Linux (using `xdotool` for window positioning), or tweak screen bounds for Windows/macOS.

## ⚙️ Setup & Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd GD_RL
   ```
2. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Setup Death Template:
   Take a screenshot of the "Attempt" text when you die in-game, crop it securely, and save it as `attempt_template.png` in the project root. This acts as the visual trigger for deaths.

## 🎮 How to Run (Training)

1. Launch Geometry Dash and ensure the window is open and unobstructed.
2. Run the main training script:
   ```bash
   python train.py
   ```
3. You have 3 seconds to tab back into the game. The AI will immediately begin capturing frames, resetting, and learning.
4. **Stopping safely**: Press `Ctrl+C` in your terminal. This triggers a `KeyboardInterrupt` which ensures the `policy_net` is safely serialized and saved to `gd_dqn_model.pth`.

## 📚 Documentation
- [Architecture Details](ARCHITECTURE.md) - Deep dive into the CNN and the Replay logic.
- [Training Guide](TRAINING_GUIDE.md) - Adjusting hyper-parameters and understanding loss metrics.

---
*Built with PyTorch, OpenCV, and lots of crashes.*
