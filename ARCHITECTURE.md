# 🏗️ Architecture Overview

This document outlines the core technical design and flow of the Geometry Dash RL Agent.

## 1. Interaction Pipeline (Game ↔ Agent)
The agent operates through an infinite loop of `Observe -> Predict -> Act -> Evaluate`.

1. **State Observation**: `screen_capture.py` uses `mss` and `xdotool` to locate the game window and grab frames at 60 FPS. These RGB frames are converted to grayscale and resized to a `128x128` matrix.
2. **State Processing**: Grayscale arrays are normalized (`0.0` to `1.0`), reshaped to `(1, 128, 128)` tensors, and passed to the PyTorch DQN.
3. **Action Prediction**: The model predicts Q-values for actions (e.g., Nothing vs Jump). It picks the max Q-value depending on the exploration rate (Epsilon).
4. **Action Execution**: `pyautogui` sends simulated physical mouse clicks to the center of the Geometry Dash window.
5. **Reward & Evaluation**: The `env.py` evaluates survival. If it detects a static screen or the "Attempt" text, it registers a massive negative penalty (`-100`). Otherwise, survival yields a `+1` reward.

## 2. Neural Network Architecture (`dqn_model.py`)

We use a Convolutional Neural Network (CNN) specifically tailored to process 2D spacial data (the screen pixels).

- **Conv Layer 1**: `(1 channel in, 32 out, kernel 8x8, stride 4)`
- **Conv Layer 2**: `(32 channels in, 64 out, kernel 4x4, stride 2)`
- **Conv Layer 3**: `(64 channels in, 64 out, kernel 3x3, stride 1)`
- **Flattening**: The `64x12x12` output tensor is flattened to `9216` nodes.
- **FC Layer 1**: Linear layer mapping `9216` to `512` with a `ReLU` activation.
- **Output Layer**: Maps `512` to `Action Size` (Default: 2: Jump vs Do Nothing).

*All activation functions in the hidden layers use ReLU (Rectified Linear Unit).*

## 3. Experience Replay & Stable Objectives

To avoid catastrophic forgetting and stabilize the target metric, we use an Experience Replay Buffer and a dual-network approach (`policy_net` and `target_net`).

- **Replay Buffer**: Stores tuples of `(state, action, reward, next_state, done_flag)`. Training batches (size `64`) are randomly sampled, breaking correlation between consecutive frames.
- **Dual Networks**: 
  - `policy_net` receives live gradients and updates every step.
  - `target_net` acts as a stable anchor for the Bellman Equation (calculating the temporal difference target). The `policy_net` weights are copied to the `target_net` every `N` steps (e.g. 50 steps).
- **Loss Calculation**: We use **Huber Loss** (`SmoothL1Loss`) coupled with strict Gradient Clipping (Max Norm = 1.0) to prevent exploding gradients when encountering massive error spikes.
