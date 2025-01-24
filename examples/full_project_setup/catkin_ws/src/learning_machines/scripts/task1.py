#!/usr/bin/env python3
import sys
import os
import json
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# Robobo import (assuming robobo_interface is in your path)
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

###############################################################################
# 1. DQN Network
###############################################################################
# class DQN(nn.Module):
#     """
#     A simple 2-layer MLP for Q-learning.
#     Input size = number of IR sensors (8).
#     Output size = number of possible discrete actions (6 in this example).
#     """
#     def __init__(self, state_size=8, action_size=8, hidden_size=64):
#         super(DQN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, action_size)
#         )
        
#     def forward(self, x):
#         return self.net(x)
class DQN(nn.Module):
    def __init__(self, state_size=8, action_size=8, hidden_size=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)

###############################################################################
# 2. Replay Buffer
###############################################################################
class ReplayBuffer:
    """A fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

###############################################################################
# 3. Select Action (Epsilon-Greedy)
###############################################################################
def select_action(dqn, state, epsilon, action_size):
    """
    Epsilon-greedy action selection.
    state: a 1D array of shape (8,) or similar
    """
    if random.random() < epsilon:
        # Explore
        return random.randint(0, action_size - 1)
    else:
        # Exploit
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = dqn(state_t)
        return torch.argmax(q_values, dim=1).item()

###############################################################################
# 4. Compute Q-Learning Loss
###############################################################################
def compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma):
    """
    Sample a batch of experiences from replay buffer, compute TD error,
    and update the network parameters via backprop.
    """
    if len(replay_buffer) < batch_size:
        return 0.0  # Not enough samples to learn
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Convert to PyTorch tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Current Q estimates
    q_values = dqn(states)
    # Only pick the Q-values corresponding to the actions taken
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q values using the target network
    with torch.no_grad():
        next_q_values = dqn_target(next_states)
        next_q_values, _ = torch.max(next_q_values, dim=1)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = nn.MSELoss()(q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

###############################################################################
# 5. Action Execution
###############################################################################
# def execute_action(rob, action):
#     """
#     Executes the given action on the robot.
#     Actions:
#       0: Move forward
#       1: Pivot left (forward-left)
#       2: Pivot right (forward-right)
#       3: Move backward
#       4: Pivot left (backward-left)
#       5: Pivot right (backward-right)
#     """
#     speed = 50
#     duration = 500

#     if action == 0:
#         # Move forward
#         rob.move_blocking(speed, speed, duration)
#     elif action == 1:
#         # Pivot left
#         rob.move_blocking(0, speed, duration)
#     elif action == 2:
#         # Pivot right
#         rob.move_blocking(speed, 0, duration)
#     elif action == 3:
#         # Move backward
#         rob.move_blocking(-speed, -speed, duration)
#     elif action == 4:
#         # Pivot left (backward)
#         rob.move_blocking(0, -speed, duration)
#     elif action == 5:
#         # Pivot right (backward)
#         rob.move_blocking(-speed, 0, duration)
def execute_action(rob, action):
    """
    Expanded set of 8 actions:
      0: Forward           (50, 50)
      1: Forward-Left      (50, 30)
      2: Forward-Right     (30, 50)
      3: Turn Left in place (50, -50)
      4: Turn Right in place (-50, 50)
      5: Backward          (-50, -50)
      6: Backward-Left     (-50, -30)
      7: Backward-Right    (-30, -50)
    """
    speed_left = 0
    speed_right = 0
    duration = 1000

    if action == 0:  # forward
        speed_left, speed_right = 50, 50
    elif action == 1:  # forward-left
        speed_left, speed_right = 50, 30
    elif action == 2:  # forward-right
        speed_left, speed_right = 30, 50
    elif action == 3:  # turn left in place
        speed_left, speed_right = 50, -50
    elif action == 4:  # turn right in place
        speed_left, speed_right = -50, 50
    elif action == 5:  # backward
        speed_left, speed_right = -50, -50
    elif action == 6:  # backward-left
        speed_left, speed_right = -50, -30
    elif action == 7:  # backward-right
        speed_left, speed_right = -30, -50

    # Execute the chosen action
    rob.move_blocking(speed_left, speed_right, duration)
###############################################################################
# 6. Reward Function
###############################################################################
def get_reward(irs, threshold, action):
    """
    Reward function based on raw IR sensor values and action:
    
    - If max(irs) > threshold, big negative reward (collision).
    - Otherwise, penalize big IR values by subtracting 0.2 * max(irs).
    - Strongly penalize backward actions (3,4,5).
    - Slight penalty for pivot (1,2).
    - Give a bonus for forward (0).
    """
    max_ir = max(irs)
    collision_penalty = -50.0
    safe_base = 1.0         # Base reward for "safe" step
    forward_bonus = 5.0     
    closeness_penalty_scale = 0.2
    
    # If collision
    if max_ir > threshold:
        return collision_penalty
    
    # Start with a base reward
    reward = safe_base
    
    # Subtract some penalty for closeness (the higher the IR, the more we penalize)
    reward -= closeness_penalty_scale * max_ir
    
    # Bonus for forward
    if action == 0:
        reward += forward_bonus
    
    # Penalize backward
    if action in [3, 4, 5]:
        reward -= 5.0

    # Penalize pivot left/right
    if action in [1, 2]:
        reward -= 2.5
    
    return reward

###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    # 7.1. Parse arguments
    # e.g., if you call `python3 script.py --simulation test`, 
    # then sys.argv will have ['script.py', '--simulation', 'test'].
    args = sys.argv[1:]  # everything after the script name
    
    if len(args) < 1:
        raise ValueError("Usage: script.py --simulation OR --hardware [test]")
    
    environment = args[0]  # e.g. '--simulation' or '--hardware'
    test_only = False
    if len(args) > 1 and args[1] == "test":
        test_only = True

    # 7.1a. Setup the environment
    if environment == "--hardware":
        rob = HardwareRobobo(camera=True)
        threshold = 40  
    elif environment == "--simulation":
        rob = SimulationRobobo()
        threshold = 100
    else:
        raise ValueError(f"{environment} is not a valid argument (use --hardware or --simulation).")

    # Hyperparameters
    state_size = 8         # 8 IR sensors
    action_size = 8        # 6 discrete actions
    max_episodes = 1000    # Training episodes
    max_steps = 50         # Steps per episode
    batch_size = 32
    gamma = 0.9
    lr = 1e-3
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 5  # Update target DQN every 5 episodes

    # Create DQN networks
    dqn = DQN(state_size, action_size)
    dqn_target = DQN(state_size, action_size)

    # Where to save/load the best model
    print(os.getcwd())

    best_model_path = "/root/catkin_ws/src/learning_machines/q_table.pt"
    # ^ Adjust if you'd prefer a different path inside the container.

    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size=10000)

    # Attempt to load existing best model (if it exists)
    model_exists = os.path.exists(best_model_path)
    if model_exists:
        print(f"Found existing model at '{best_model_path}'. Loading...")
        dqn.load_state_dict(torch.load(best_model_path))
        dqn_target.load_state_dict(dqn.state_dict())
        dqn.eval()
        print("Loaded best model successfully!")
    else:
        print(f"No existing model found at: {best_model_path}. Starting from scratch.")

    # Synchronize target network
    dqn_target.load_state_dict(dqn.state_dict())
    
    # 7.2. If we only want to test, skip training
    if test_only:
        print("\n--- Testing the Best Learned Policy (skip training) ---")
        dqn.eval()  # ensure we're in eval mode

        test_episodes = 100  # how many test episodes you want
        test_steps = 50      # steps per test episode
        test_epsilon = 0.0   # purely exploit

        for test_ep in range(test_episodes):
            print(f"\nTest Episode {test_ep+1}")

            for step in range(test_steps):
                # 1) Read IR sensors => state
                irs = rob.read_irs()
                state = np.array(irs, dtype=np.float32)

                # 2) Select action (pure exploitation)
                with torch.no_grad():
                    if random.random() < test_epsilon:
                        action = random.randint(0, action_size - 1)
                    else:
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        q_values = dqn(state_t)
                        action = torch.argmax(q_values, dim=1).item()

                # 3) Execute action
                execute_action(rob, action)

                # 4) Optionally check near-collision
                max_ir = max(rob.read_irs())
                if max_ir > (1.5 * threshold):
                    print("Test run near-collision. Stopping test episode.")
                    break

        print("\nTest run complete! Exiting now.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Otherwise, do the normal training routine below.
    # -------------------------------------------------------------------------
    epsilon = epsilon_start
    best_score = float('-inf')
    all_sensor_data = []

    ###############################################################################
    # 7.3. Training Loop
    ###############################################################################
    for episode in range(max_episodes):
        print(f"\n--- Training Episode {episode+1} ---")
        episode_sensor_data = []
        episode_reward = 0.0  # track sum of rewards this episode

        for step in range(max_steps):
            # 1) Read IR sensors => state
            irs = rob.read_irs()  # e.g. [irs0, irs1, ... irs7]
            state = np.array(irs, dtype=np.float32)
            episode_sensor_data.append(irs)

            # 2) Select action with epsilon-greedy
            action = select_action(dqn, state, epsilon, action_size)

            # 3) Execute action
            execute_action(rob, action)

            # 4) Observe new state
            next_irs = rob.read_irs()
            next_state = np.array(next_irs, dtype=np.float32)

            # 5) Calculate reward
            reward = get_reward(next_irs, threshold, action)
            episode_reward += reward

            # 6) Check if done (end episode early if near collision)
            max_ir = max(next_irs)
            done = 1.0 if max_ir > (1.2 * threshold) else 0.0

            # 7) Store transition
            replay_buffer.add(state, action, reward, next_state, done)

            # 8) Train the DQN
            loss = compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma)

            # 9) Update state
            state = next_state

            # 10) Break if done
            if done == 1.0:
                print(f"Near-collision detected - ending episode {episode+1} early.")
                break

        all_sensor_data.append(episode_sensor_data)
        
        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            dqn_target.load_state_dict(dqn.state_dict())
            print(f"Target network updated at episode {episode+1}")

        # Check if this is the best model so far
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save(dqn.state_dict(), best_model_path)
            print(f"New best model saved with episode reward {episode_reward:.2f} at '{best_model_path}'")

    # Save sensor data after training
    output_file = "sensor_data_dqn.json"
    with open(output_file, "w") as f:
        json.dump(all_sensor_data, f, indent=4)
    print(f"\nTraining data saved to {output_file}")

    print("\nTraining run complete!")
    print(f"Best model is at: {best_model_path}")

    ###############################################################################
    # 7.4. Automatic Test after Training (final part)
    # ############################################################################
    print("\n--- Automatic Testing after Training ---")
    dqn.eval()  # go into eval mode
    # Reload best model just to be sure
    if os.path.exists(best_model_path):
        dqn.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from '{best_model_path}' for final test.")
    else:
        print("No best model found. Using current model for test.")

    test_episodes = 10  # a shorter test after training
    test_steps = 50
    test_epsilon = 0.0

    for test_ep in range(test_episodes):
        print(f"\nTest Episode (Post-Training) {test_ep+1}")
        for step in range(test_steps):
            irs = rob.read_irs()
            state = np.array(irs, dtype=np.float32)

            with torch.no_grad():
                if random.random() < test_epsilon:
                    action = random.randint(0, action_size - 1)
                else:
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = dqn(state_t)
                    action = torch.argmax(q_values, dim=1).item()

            execute_action(rob, action)
            max_ir = max(rob.read_irs())
            if max_ir > (1.5 * threshold):
                print("Test run near-collision. Stopping test episode.")
                break

    print("\nAll done! Exiting now.")
