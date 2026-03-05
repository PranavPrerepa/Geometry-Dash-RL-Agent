# 🧠 Training Guide & Hyper parameters

Training RL Agents from raw pixels is notoriously unstable and requires careful hyper-parameter tuning. This guide explains the core configurations in `train.py`.

## Hyperparameters

- **`BATCH_SIZE (64)`**: Number of experiences sampled from the replay buffer per optimization step.
- **`GAMMA (0.99)`**: Discount factor. A high gamma means the agent cares deeply about long-term rewards (beating the level) rather than just immediate survival.
- **`EPSILON_START (1.0)`**: The agent starts completely blind, taking 100% random actions to explore the environment state space.
- **`EPSILON_END (0.05)`**: The permanent 5% randomness leftover at the end of training to ensure the agent doesn't get stuck in a strict local minimum.
- **`EPSILON_DECAY (10000)`**: Controls how fast the exploration decays. An exponential delay is used.
- **`LEARNING_RATE (1e-4)`**: Standard low learning rate for Adam optimizer to ensure stable updates on the Huber Loss function.
- **`TARGET_UPDATE (50)`**: How often the stable Target Network copies weights from the actively training Policy Network.
- **`MEMORY_CAPACITY (20000)`**: How many experiences to hold before overwriting old ones.

## Visual Logs

During training, the script outputs state info:

```text
Ep   1 | Step   42 | Action: Jump  | Reward:    1.0 | Loss: 0.0412
Ep   1 | Step   43 | Action: None  | Reward: -100.0 | Loss: 0.8111
Episode 1 | Steps: 43 | Reward: -57 | Loss: 0.8111 | Epsilon: 0.985
```

- **Loss** will inevitably spike when the agent encounters an unpredictable death penalty.
- Over time, you should observe the agent surviving longer (`Steps` increasing per episode).

## Handling Game Resets

The `env.py` manages dying by executing a macro (clicking the retry button location) and pressing the space bar. Keep the game window in a static location to prevent coordinate drifting between deaths.

## Saving & Resuming

Pressing `Ctrl+C` terminates the execution safely by bypassing the Exception and dumping the PyTorch weights to `gd_dqn_model.pth`. You can load this file manually to resume training in future iterations.
