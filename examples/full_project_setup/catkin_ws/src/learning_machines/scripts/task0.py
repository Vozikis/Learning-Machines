#!/usr/bin/env python3
# import sys
# import json
# from robobo_interface import SimulationRobobo, HardwareRobobo
# import cv2
# import torch
# import sys
# import json
# import torch
# import random
# import numpy as np
# from collections import deque
# from robobo_interface import SimulationRobobo, HardwareRobobo

# from robobo_interface import (
#     IRobobo,
#     Emotion,
#     LedId,
#     LedColor,
#     SoundEmotion,
#     SimulationRobobo,
#     HardwareRobobo,
# )
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         raise ValueError(
#             """To run, we need to know if we are running on hardware of simulation
#             Pass `--hardware` or `--simulation` to specify."""
#         )
#     elif sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=True)
#          # Define proximity threshold for hardware
#         threshold = 50


#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#         # Define proximity threshold for simulation
#         threshold = 150

#     else:
#         raise ValueError(f"{sys.argv[1]} is not a valid argument.")

#     # Initialize variables
#     repetitions = 5
#     obstacle_detections_per_repetition = 100
#     all_sensor_data = []  # To store sensor data across repetitions

#     for repetition in range(repetitions):
#         print(f"Starting repetition {repetition + 1}")
#         obstacle_detections = 0
#         sensor_data_per_repetition = []

#         while obstacle_detections < obstacle_detections_per_repetition:
#             # Read sensor data
#             irs = rob.read_irs()
#             print("IRS readings:", irs)


#             obstacle_detected = any(value > threshold for value in irs)

#             if obstacle_detected:
#                 print("Obstacle detected! Avoiding_tsouf_tsouf")

#                 # Save sensor data for this detection
#                 sensor_data_per_repetition.append(irs)
#                 obstacle_detections += 1

#                 # Handle obstacle avoidance
#                 if any(value > threshold for value in irs[2:6] or irs[7]):
#                     rob.move_blocking(-25, -25, 1000)  # Move backward for 1 second
#                 else:
#                     rob.move_blocking(25, 25, 1000)  # Move forward slightly

#                 if irs[7] >= irs[5]:
#                     rob.move_blocking(25, 0, 1000)  # Pivot right for 1 second
#                 else:
#                     rob.move_blocking(0, 25, 1000)

#                 rob.sleep(0.5)  # Allow time for new sensor readings

#             else:
#                 rob.move_blocking(25, 25, 1000)  # Move forward for 1 second

#         # Save sensor data for this repetition
#         all_sensor_data.append(sensor_data_per_repetition)
#         print(f"Finished repetition {repetition + 1}")

#     # Save all data to a JSON file
#     output_file = "sensor_data.json"
#     with open(output_file, "w") as f:
#         json.dump(all_sensor_data, f, indent=4)
#     print(f"Sensor data saved to {output_file}")



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
class DQN(nn.Module):
    """
    A simple 2-layer MLP for Q-learning.
    Input size = number of IR sensors (8).
    Output size = number of possible discrete actions (6 in this example).
    """
    def __init__(self, state_size=8, action_size=6, hidden_size=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
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
def execute_action(rob, action):
    """
    Executes the given action on the robot.
    Actions:
      0: Move forward
      1: Pivot left (forward-left)
      2: Pivot right (forward-right)
      3: Move backward
      4: Pivot left (backward-left)
      5: Pivot right (backward-right)
    """
    speed = 25
    duration = 500

    if action == 0:
        # Move forward
        rob.move_blocking(speed, speed, duration)
    elif action == 1:
        # Pivot left
        rob.move_blocking(0, speed, duration)
    elif action == 2:
        # Pivot right
        rob.move_blocking(speed, 0, duration)
    elif action == 3:
        # Move backward
        rob.move_blocking(-speed, -speed, duration)
    # elif action == 4:
    #     # Pivot left (backward)
    #     rob.move_blocking(0, -speed, duration)
    # elif action == 5:
    #     # Pivot right (backward)
    #     rob.move_blocking(-speed, 0, duration)

###############################################################################
# 6. Reward Function
###############################################################################
def get_reward(irs, threshold, action):
    """
    Reward function based on raw IR sensor values and action:
    
    - If max(irs) > threshold, big negative reward (collision).
    - Otherwise, penalize big IR values by subtracting 0.01 * max(irs).
    - Strongly penalize backward actions (3,4,5).
    - Give a slight bonus for forward action (0).
    """
    max_ir = max(irs)
    collision_penalty = -50.0
    safe_base = 2         # Base reward for staying "safe"
    forward_bonus = 5     # Extra reward for forward
    closeness_penalty_scale = 0.1
    
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
    
    # **Penalize backward** actions heavily
    if action in [3]:
        reward -= 2.5
    if action in [2, 1]:
        reward -= 1.0
    return reward







###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    # 7.1. Initialization
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation.
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        threshold = 40  
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        threshold = 100
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # Hyperparameters
    state_size = 8         # 8 IR sensors
    action_size = 6        # 6 discrete actions
    max_episodes = 500    # Training episodes
    max_steps = 50         # Steps per episode
    batch_size = 32
    gamma = 0.99
    lr = 1e-3
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1
    target_update_freq = 5  # Update target DQN every 5 episodes

    # Create DQN networks
    dqn = DQN(state_size, action_size)
    dqn_target = DQN(state_size, action_size)

    # -------------------------------------------------------------------------
    # 7.1a. Load existing policy if available
    # -------------------------------------------------------------------------
    model_path = "best_dqn.pth"
    if os.path.exists(model_path):
        print(f"Found existing model '{model_path}'. Loading it...")
        dqn.load_state_dict(torch.load(model_path))
        dqn_target.load_state_dict(dqn.state_dict())
        print("Model loaded successfully!")
    else:
        print("No existing model found. Starting training from scratch.")

    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size=10000)

    # Synchronize target network
    dqn_target.load_state_dict(dqn.state_dict())
    
    epsilon = epsilon_start
    
    all_sensor_data = []

    ###############################################################################
    # 7.2. Training Loop
    ###############################################################################
    for episode in range(max_episodes):
        print(f"\n--- Training Episode {episode+1} ---")
        episode_sensor_data = []
        print(epsilon)

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

            # 6) Check if done (end episode early if near collision)
            max_ir = max(next_irs)
            done = 1.0 if max_ir > (1 * threshold) else 0.0

            # 7) Store transition
            replay_buffer.add(state, action, reward, next_state, done)

            # 8) Train the DQN
            loss = compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma)

            # 9) Update state
            state = next_state

            # 10) Break if done
            # if done == 1.0:
            #     print("Near-collision detected - ending episode early.")
            #     break

        all_sensor_data.append(episode_sensor_data)
        
        # Decay epsilon
        epsilon = max(1- episode/max_episodes  , epsilon_end)

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            dqn_target.load_state_dict(dqn.state_dict())
            print(f"Target network updated at episode {episode+1}")
        print(reward)

    # Save sensor data after training
    output_file = "sensor_data_dqn.json"
    with open(output_file, "w") as f:
        json.dump(all_sensor_data, f, indent=4)
    print(f"\nTraining data saved to {output_file}")

    # -------------------------------------------------------------------------
    # 7.2a. Save the final trained policy
    # -------------------------------------------------------------------------
    torch.save(dqn.state_dict(), model_path)
    print(f"Final trained policy saved to {model_path}")

    ###############################################################################
    # 7.3. Testing Loop (Best Learned Policy)
    ###############################################################################
    print("\n--- Testing the Best Learned Policy ---")
    test_episodes = 100   # how many test episodes you want
    test_steps = 50        # steps per test episode
    
    # We set epsilon to 0 so there's no random exploration
    test_epsilon = 0.0

    for test_ep in range(test_episodes):
        print(f"\nTest Episode {test_ep+1}")
        # (Optional) reset or reposition the robot for a fresh environment

        for step in range(test_steps):
            # 1) Read IR sensors => state
            irs = rob.read_irs()
            state = np.array(irs, dtype=np.float32)

            # 2) Select action with epsilon=0 (pure exploitation)
            action = select_action(dqn, state, test_epsilon, action_size)

            # 3) Execute action
            execute_action(rob, action)

            # 4) Check if near collision => can break if you want
            max_ir = max(rob.read_irs())
            if max_ir > (1 * threshold):
                print("Test run near-collision. Stopping test episode.")
                break

    print("Test run complete!")



# #!/usr/bin/env python3
# import sys
# import os
# import json
# import random
# import numpy as np
# from collections import deque

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Robobo import (assuming robobo_interface is in your path)
# from robobo_interface import (
#     IRobobo,
#     Emotion,
#     LedId,
#     LedColor,
#     SoundEmotion,
#     SimulationRobobo,
#     HardwareRobobo,
# )

# ###############################################################################
# # 1. DQN Network
# ###############################################################################
# class DQN(nn.Module):
#     def __init__(self, state_size=8, action_size=6, hidden_size=64):
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

# ###############################################################################
# # 2. Replay Buffer
# ###############################################################################
# class ReplayBuffer:
#     def __init__(self, buffer_size=10000):
#         self.buffer = deque(maxlen=buffer_size)
    
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size=64):
#         samples = random.sample(self.buffer, k=batch_size)
#         states, actions, rewards, next_states, dones = zip(*samples)
#         return states, actions, rewards, next_states, dones
    
#     def __len__(self):
#         return len(self.buffer)

# ###############################################################################
# # 3. Select Action (Epsilon-Greedy)
# ###############################################################################
# def select_action(dqn, state, epsilon, action_size):
#     if random.random() < epsilon:
#         return random.randint(0, action_size - 1)
#     else:
#         state_t = torch.FloatTensor(state).unsqueeze(0)
#         q_values = dqn(state_t)
#         return torch.argmax(q_values, dim=1).item()

# ###############################################################################
# # 4. Compute Q-Learning Loss
# ###############################################################################
# def compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma):
#     if len(replay_buffer) < batch_size:
#         return 0.0
    
#     states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
#     states = torch.FloatTensor(states)
#     actions = torch.LongTensor(actions)
#     rewards = torch.FloatTensor(rewards)
#     next_states = torch.FloatTensor(next_states)
#     dones = torch.FloatTensor(dones)

#     q_values = dqn(states)
#     q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
#     with torch.no_grad():
#         next_q_values = dqn_target(next_states)
#         next_q_values, _ = torch.max(next_q_values, dim=1)
#         target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
#     loss = nn.MSELoss()(q_values, target_q_values)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     return loss.item()

# ###############################################################################
# # 5. Action Execution
# ###############################################################################
# def execute_action(rob, action):
#     speed = 25
#     duration = 500

#     if action == 0:
#         rob.move_blocking(speed, speed, duration)
#     elif action == 1:
#         rob.move_blocking(0, speed, duration)
#     elif action == 2:
#         rob.move_blocking(speed, 0, duration)
#     elif action == 3:
#         rob.move_blocking(-speed, -speed, duration)

# ###############################################################################
# # 6. Reward Function and Collision Check
# ###############################################################################
# def check_collision(irs, threshold):
#     special_sensors = [0, 4, 5, 7]
#     for i, val in enumerate(irs):
#         if i in special_sensors:
#             if val > (0.8 * threshold):
#                 return True
#         else:
#             if val > threshold:
#                 return True
#     return False

# def get_reward(irs, threshold, action):
#     collision_penalty = -50.0
#     safe_base = 2
#     forward_bonus = 5
#     reward = 0
#     # closeness_penalty_scale = 0.1

#     if check_collision(irs, threshold):
#         reward-= collision_penalty

#     max_ir = max(irs)
#     reward = safe_base
#     # reward -= closeness_penalty_scale * max_ir

#     if action == 0:
#         reward += forward_bonus
#     if action == 3:
#         reward -= 2.5
#     if action in [2, 1]:
#         reward -= 1.0
#     print(f"Reward: {reward}", "max_ir", max_ir)
#     return reward

# ###############################################################################
# # 7. Main
# ###############################################################################
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         raise ValueError("Pass `--hardware` or `--simulation` to specify.")
#     elif sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=True)
#         threshold = 40
#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#         threshold = 150
#     else:
#         raise ValueError(f"{sys.argv[1]} is not a valid argument.")

#     state_size = 8
#     action_size = 6
#     max_episodes = 200
#     max_steps = 50
#     batch_size = 32
#     gamma = 0.99
#     lr = 1e-3
#     epsilon_start = 1.0
#     epsilon_end = 0.01
#     epsilon_decay = 0.995
#     target_update_freq = 5

#     dqn = DQN(state_size, action_size)
#     dqn_target = DQN(state_size, action_size)
    
#     model_path = "best_dqn.pth"
#     if os.path.exists(model_path):
#         print(f"Found existing model '{model_path}'. Loading it...")
#         dqn.load_state_dict(torch.load(model_path))
#         dqn_target.load_state_dict(dqn.state_dict())
#     else:
#         print("No existing model found. Starting training from scratch.")

#     optimizer = optim.Adam(dqn.parameters(), lr=lr)
#     replay_buffer = ReplayBuffer(buffer_size=10000)
#     dqn_target.load_state_dict(dqn.state_dict())

#     epsilon = epsilon_start
#     all_sensor_data = []

#     for episode in range(max_episodes):
#         print(f"\n--- Training Episode {episode+1} ---")
#         episode_sensor_data = []

#         for step in range(max_steps):
#             irs = rob.read_irs()
#             state = np.array(irs, dtype=np.float32)
#             episode_sensor_data.append(irs)

#             action = select_action(dqn, state, epsilon, action_size)
#             execute_action(rob, action)

#             next_irs = rob.read_irs()
#             next_state = np.array(next_irs, dtype=np.float32)

#             reward = get_reward(next_irs, threshold, action)
#             done = 0.0  # Ensure the episode doesn't end on collision


#             replay_buffer.add(state, action, reward, next_state, done)
#             loss = compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma)

#             state = next_state

#         all_sensor_data.append(episode_sensor_data)
#         epsilon = max(epsilon * epsilon_decay, epsilon_end)

#         if (episode + 1) % target_update_freq == 0:
#             dqn_target.load_state_dict(dqn.state_dict())

#     output_file = "sensor_data_dqn.json"
#     with open(output_file, "w") as f:
#         json.dump(all_sensor_data, f, indent=4)

#     torch.save(dqn.state_dict(), model_path)

#     print("\n--- Testing the Best Learned Policy ---")
#     test_episodes = 200
#     test_steps = 50
#     test_epsilon = 0.0

#     for test_ep in range(test_episodes):
#         for step in range(test_steps):
#             irs = rob.read_irs()
#             state = np.array(irs, dtype=np.float32)
#             action = select_action(dqn, state, test_epsilon, action_size)
#             execute_action(rob, action)

#             if check_collision(rob.read_irs(), threshold):
#                 break

#     print("Test run complete!")




