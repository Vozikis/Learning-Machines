#!/usr/bin/env python3
import sys
import os
import random
import time
import numpy as np
from collections import deque

import cv2
import torch
import torch.nn as nn
import torch.optim as optim

###############################################################################
# 0. Robobo Interface
###############################################################################
from robobo_interface import (
    SimulationRobobo,
    HardwareRobobo,
)


###############################################################################
# 1. DQN Network
###############################################################################
class DQN(nn.Module):
    def __init__(self, state_size=6, action_size=3, hidden_size=64):
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
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        samples = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones
        )

    def __len__(self):
        return len(self.buffer)


###############################################################################
# 3. Select Action (Epsilon-Greedy)
###############################################################################
def select_action(dqn, state, epsilon, action_size):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = dqn(state_t)
        return torch.argmax(q_values, dim=1).item()


###############################################################################
# 4. Compute TD Loss
###############################################################################
def compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Current Q-values
    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q-values (from target network)
    with torch.no_grad():
        next_q_values = dqn_target(next_states).max(dim=1)[0]
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
    Actions:
      0: Forward            (speed_left=90,  speed_right=90)
      1: Turn Left in place (speed_left=10,  speed_right=-10)
      2: Turn Right in place(speed_left=-10, speed_right=10)
    """
    duration = 1000  # milliseconds

    if action == 0:  # forward
        speed_left, speed_right = 90, 90
    elif action == 1:  # turn left
        speed_left, speed_right = 10, -10
    elif action == 2:  # turn right
        speed_left, speed_right = -10, 10

    rob.move_blocking(speed_left, speed_right, duration)


###############################################################################
# 6. Camera-Based Detection
###############################################################################
def detect_green_boxes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            green_boxes.append(((x, y), (w, h)))
    return green_boxes

def detect_red_boxes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            red_boxes.append(((x, y), (w, h)))
    return red_boxes


###############################################################################
# 7. Create State
#    state = [IR4, IR5, IR6, IR7, red_detected, green_detected]
###############################################################################
def create_state(rob):
    # Read IR sensors
    irs = rob.read_irs()
    s4, s5, s6, s7 = irs[4], irs[5], irs[6], irs[7]

    # Detect red and green in camera
    red_flag, green_flag = 0.0, 0.0
    frame = rob.read_image_front()
    if frame is not None:
        if len(detect_red_boxes(frame)) > 0:
            red_flag = 1.0
        if len(detect_green_boxes(frame)) > 0:
            green_flag = 1.0

    # Return as float32 array
    return np.array([s4, s5, s6, s7, red_flag, green_flag], dtype=np.float32)


def compute_reward(action, old_state, new_state, threshold):
    """
    new_state layout: [IR4, IR5, IR6, IR7, red_detected, green_detected]
    """
    done = False
    reward = 0.0

    # Unpack
    ir4_old = old_state[0]
    ir4_new = new_state[0]
    ir5, ir6, ir7 = new_state[1], new_state[2], new_state[3]
    red_flag = new_state[4]
    green_flag = new_state[5]

    # 1) Small living cost
    reward -= 0.1

    # 2) Collision with sides
    if ir5 > threshold or ir6 > threshold or ir7 > threshold:
        reward -= 20.0
        done = True
        return reward, done

    # 3) Reward for seeing red
    if red_flag == 1.0:
        reward += 2.5

    # 4) Check if we are pushing the red box
    pushing_red = (ir4_new > threshold)
    if pushing_red:
        reward += 5.0  # general pushing reward
        if green_flag == 1.0:
            # Large success reward
            reward += 20.0
            done = False
    else:
        # Not pushing the red box => small penalty
        reward -= 0.5

    # 5) Minor shaping for forward vs turn
    if action == 0:  # forward
        reward += 0.25
    elif action in [1, 2]:  # turning
        reward -= 0.1

    return reward, done


###############################################################################
# 9. Main
###############################################################################
if __name__ == "__main__":
    args = sys.argv[1:]
    save_path = "/root/results/q_table.pt"

    if len(args) < 1:
        raise ValueError("Usage: script.py --simulation OR --hardware [train or test]")

    environment = args[0]
    mode = args[1] if len(args) > 1 else "train"

    # Decide environment
    if environment == "--hardware":
        rob = HardwareRobobo()
        threshold = 15  # IR threshold (real robot)
    elif environment == "--simulation":
        rob = SimulationRobobo()
        threshold = 85  # IR threshold (simulator)
    else:
        raise ValueError(f"Invalid argument: {environment}")

    # Hyperparams
    state_size = 6
    action_size = 3  # forward, turn-left, turn-right
    hidden_size = 64
    lr = 1e-3
    gamma = 0.95
    batch_size = 32
    replay_buffer_size = 10000

    max_episodes = 150
    steps_per_episode = 100
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.1

    # Build DQNs, optimizer, replay buffer
    dqn = DQN(state_size, action_size, hidden_size)
    dqn_target = DQN(state_size, action_size, hidden_size)
    dqn_target.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size)

    episode_rewards = []

    ############################################################################
    # TRAIN MODE
    ############################################################################
    if mode == "train":
        for episode in range(max_episodes):
            # Reset environment
            rob.stop_simulation()
            time.sleep(0.5)
            rob.play_simulation()
            time.sleep(0.5)

            # Optionally tilt camera (depends on your environment)
            rob.set_phone_tilt(109, 80)
            time.sleep(1.0)
            print(f"\n[EPISODE {episode+1}] Environment reset.")

            # Init state
            state = create_state(rob)
            total_reward = 0.0

            for step in range(steps_per_episode):
                # Epsilon-greedy action
                action = select_action(dqn, state, epsilon, action_size)
                # Execute
                execute_action(rob, action)
                # Next state
                next_state = create_state(rob)

                # Reward & done
                reward, done = compute_reward(action, state, next_state, threshold)
                total_reward += reward

                # Store in replay
                replay_buffer.add(state, action, reward, next_state, done)
                # Train step
                loss_val = compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma)

                # ------------------------------
                # PRINT for logging/plotting
                # ------------------------------
                print(
                    f"Step={step+1}, "
                    f"Loss={loss_val:.4f}, "
                    f"Action={action}, "
                    f"Reward={reward:.2f}, "
                    f"TotalReward={total_reward:.2f}, "
                    f"Epsilon={epsilon:.3f}, "
                    f"IR4={next_state[0]:.2f}, IR5={next_state[1]:.2f}, "
                    f"IR6={next_state[2]:.2f}, IR7={next_state[3]:.2f}, "
                    f"Red={int(next_state[4])}, Green={int(next_state[5])}"
                )
                # ------------------------------

                state = next_state

                if done:
                    print(f"  [EPISODE {episode+1}] Done at step={step+1} (reason=done).")
                    break
            else:
                print(f"  [EPISODE {episode+1}] Reached step limit {steps_per_episode} (no done).")

            # Epsilon decay & target update
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            dqn_target.load_state_dict(dqn.state_dict())

            # Log
            episode_rewards.append(total_reward)
            print(f"  [EPISODE {episode+1}] Total Reward: {total_reward:.2f}")

            # Every 10 episodes, print average of the last 10
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"[INFO] Ep {episode+1}/{max_episodes}, "
                      f"Avg(Last10)={avg_reward:.2f}, Epsilon={epsilon:.3f}")

            # Save after each episode (or modify frequency as needed)
            torch.save(dqn.state_dict(), save_path)
            print(f"[TRAIN] Model saved to {save_path}")

    ############################################################################
    # TEST MODE
    ############################################################################
    elif mode == "test":
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model file '{save_path}' not found. Train first.")
        dqn.load_state_dict(torch.load(save_path))
        print(f"[TEST] Loaded model from {save_path}")

        # Simple test runs
        test_episodes = 5
        for e in range(test_episodes):
            rob.stop_simulation()
            time.sleep(1.0)
            rob.play_simulation()
            time.sleep(1.0)

            rob.set_phone_tilt(109, 80)
            time.sleep(1.0)
            print(f"[TEST EPISODE {e+1}]")

            state = create_state(rob)
            for step in range(50):
                # Nearly greedy in test
                action = select_action(dqn, state, 0.05, action_size)
                execute_action(rob, action)
                next_state = create_state(rob)

                _, done = compute_reward(action, state, next_state, threshold)
                state = next_state

                if done:
                    print(f"[TEST EPISODE {e+1}] Done at step={step+1}.")
                    break
            else:
                print(f"[TEST EPISODE {e+1}] Step limit reached.")
