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

from robobo_interface import (
    SimulationRobobo,
    HardwareRobobo,
)

###############################################################################
# 1. DQN Network
###############################################################################
class DQN(nn.Module):
    # Change default state_size to 6 (IR[4], IR[5], IR[6], IR[7], detected_flag, center_offset)
    def __init__(self, state_size=6, action_size=4, hidden_size=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
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
        return np.array(states), actions, rewards, np.array(next_states), dones

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

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

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
    0: Forward            (fast)
    1: Turn Left in place
    2: Turn Right in place
    3: Backward
    """
    speed_left = 0
    speed_right = 0
    duration = 1000

    # Increase forward speed to move quickly
    if action == 0:  # forward
        speed_left, speed_right = 70, 70
    elif action == 1:  # turn left in place
        speed_left, speed_right = 17, -17
    elif action == 2:  # turn right in place
        speed_left, speed_right = -17, 17
    elif action == 3:  # backward
        speed_left, speed_right = -50, -50

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
        if area > 250:  # only consider bigger areas
            x, y, w, h = cv2.boundingRect(cnt)
            green_boxes.append(((x, y), (w, h)))
    return green_boxes, mask_clean

def get_largest_box_area(frame):
    if frame is None:
        return 0.0
    green_boxes, _ = detect_green_boxes(frame)
    if not green_boxes:
        return 0.0
    largest_box = max(green_boxes, key=lambda b: b[1][0] * b[1][1])
    (x, y), (w, h) = largest_box
    return float(w * h)

def get_center_offset(frame):
    """
    Returns a float in [-1, 1] indicating how far the largest green box
    is from the horizontal center of the image. 0.0 means perfectly centered.
    """
    if frame is None:
        return 0.0

    h_img, w_img = frame.shape[:2]
    green_boxes, _ = detect_green_boxes(frame)
    if not green_boxes:
        return 0.0

    # Choose the "closest" box by largest area (usually the biggest means closer)
    largest_box = max(green_boxes, key=lambda b: b[1][0] * b[1][1])
    (x, y), (w, h) = largest_box
    box_center_x = x + w / 2.0

    # Normalize offset to [-1, 1]
    center_x = w_img / 2.0
    offset = (box_center_x - center_x) / (w_img / 2.0)
    return float(offset)


###############################################################################
# 7. Create State
###############################################################################
def create_state(rob):
    """
    State:
      IR sensors [4,5,6,7]
      detected_flag (1 if any green box seen, else 0)
      center_offset (for largest green box)
    => 6D state vector
    """
    irs = rob.read_irs()
    state_irs = np.array([irs[4], irs[5], irs[6], irs[7]])

    frame = rob.read_image_front()
    if frame is not None:
        green_boxes, _ = detect_green_boxes(frame)
        detected_flag = 1.0 if green_boxes else 0.0
        offset = get_center_offset(frame)
    else:
        detected_flag = 0.0
        offset = 0.0

    state = np.concatenate((state_irs, [detected_flag, offset]))
    return state


###############################################################################
# 8. COMBINED Reward Function (Modified for Fast Approach + Collision Avoidance)
###############################################################################
def compute_reward_combined(
    action,
    old_area,
    new_area,
    old_food_count,
    new_food_count,
    old_state,
    new_state,
    collision_threshold
):
    """
    Priorities:
      1) Move quickly toward the box => bigger positive for forward movement & area increase
      2) Avoid collisions => heavy penalty if IR sensors exceed threshold
    """
    reward = 0.0

    # (A) Small living cost (slightly negative but not too large)
    reward -= 0.05

    # (B) Collision penalty with obstacles (non-food) - strong penalty
    if max(new_state[:4]) > collision_threshold:
        reward -= 50.0  # Increased penalty

    # (C) Camera-based shaping: if area increased => significant positive, else small negative
    if new_area > old_area:
        reward += 5.0
    else:
        reward -= 10.0

    # (D) Big reward for each newly collected food
    foods_gained = new_food_count - old_food_count
    if foods_gained > 0:
        reward += 100.0 * foods_gained

    # (E) Optional shaping by action
    #     Encourage forward movement strongly, penalize backward
    if action == 0:        # forward
        reward += 1.0
    elif action in [1, 2]: # turn left/right
        reward -= 0.05
    else:                  # backward
        reward -= 3.0

    # (F) Extra shaping to encourage centering the box quickly
    # new_state[-1] is 'offset' in [-1, 1]
    offset_now = abs(new_state[-1])
    # Give up to +1.0 if perfectly centered, 0 if offset=1
    reward += (1.0 - offset_now)

    return reward


###############################################################################
# 9. TEST MODE (3min)
###############################################################################
def test_model_3min(dqn, rob, action_size, threshold=200, run_time=180, success_area=5000):
    dqn.eval()  # disable training layers like dropout
    
    rob.set_phone_tilt(100, 30)
    start_time = time.time()
    box_counter = 0

    # Initialize state + camera
    state = create_state(rob)
    frame = rob.read_image_front()
    old_area = get_largest_box_area(frame)

    while (time.time() - start_time) < run_time:
        # Greedy action
        action = select_action(dqn, state, epsilon=0.0, action_size=action_size)
        execute_action(rob, action)

        # Next state + camera area
        next_state = create_state(rob)
        frame_next = rob.read_image_front()
        new_area = get_largest_box_area(frame_next)

        # Check if area surpasses success_area => "touched" visually
        if new_area > success_area:
            box_counter += 1
            print(f"[TEST] Box touched visually! count={box_counter}")

        # Update
        state = next_state
        old_area = new_area

    print(f"[TEST] 3 min test done. Boxes touched (visually): {box_counter}")


###############################################################################
# 10. Main
###############################################################################
if __name__ == "__main__":
    args = sys.argv[1:]
    save_path = "/root/results/q_table.pt"

    if len(args) < 1:
        raise ValueError("Usage: script.py --simulation OR --hardware [train or test]")

    environment = args[0]  # e.g. --simulation or --hardware
    mode = args[1] if len(args) > 1 else "train"

    # Decide environment
    if environment == "--hardware":
        rob = HardwareRobobo()
        threshold = 15
    elif environment == "--simulation":
        rob = SimulationRobobo()
        threshold = 200
    else:
        raise ValueError(f"Invalid argument: {environment}")

    # Hyperparams
    state_size = 6
    action_size = 4
    hidden_size = 64
    lr = 1e-3
    gamma = 0.9
    batch_size = 32
    replay_buffer_size = 10000

    max_episodes = 500
    steps_per_episode = 100
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    # Build DQNs, optimizer, replay buffer
    dqn = DQN(state_size, action_size, hidden_size)
    dqn_target = DQN(state_size, action_size, hidden_size)
    dqn_target.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size)

    episode_rewards = []

    # -------------------------------------------------------------------------
    # TRAIN MODE
    # -------------------------------------------------------------------------
    if mode == "train":
        food_target = 7  # end episode if environment-based script says 7 items

        for episode in range(max_episodes):
            # Reset sim each episode
            rob.stop_simulation()
            time.sleep(1.0)
            
            rob.play_simulation()
            time.sleep(1.0)

            rob.set_phone_tilt(100, 30)
            print(f"[EPISODE {episode+1}] Env reset. Tilt set to (100,30).")

            state = create_state(rob)
            frame = rob.read_image_front()
            old_area = get_largest_box_area(frame)

            old_food_count = rob.get_nr_food_collected()
            total_reward = 0.0

            reason_for_done = None

            for step in range(steps_per_episode):
                action = select_action(dqn, state, epsilon, action_size)
                execute_action(rob, action)

                next_state = create_state(rob)
                frame_next = rob.read_image_front()
                new_area = get_largest_box_area(frame_next)

                new_food_count = rob.get_nr_food_collected()

                reward = compute_reward_combined(
                    action=action,
                    old_area=old_area,
                    new_area=new_area,
                    old_food_count=old_food_count,
                    new_food_count=new_food_count,
                    old_state=state,
                    new_state=next_state,
                    collision_threshold=threshold
                )
                total_reward += reward

                done = False
                if new_food_count >= food_target:
                    done = True
                    reason_for_done = f"collected {food_target} foods"

                replay_buffer.add(state, action, reward, next_state, done)
                compute_td_loss(dqn, dqn_target, optimizer, replay_buffer, batch_size, gamma)

                state = next_state
                old_area = new_area
                old_food_count = new_food_count

                if done:
                    print(f"  [EPISODE {episode+1}] Done at step={step+1} => {reason_for_done}")
                    break
            else:
                reason_for_done = "step limit reached"
                print(f"  [EPISODE {episode+1}] Done at step={steps_per_episode} => {reason_for_done}")

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            dqn_target.load_state_dict(dqn.state_dict())

            episode_rewards.append(total_reward)
            print(f"  [EPISODE {episode+1}] Total Reward: {total_reward:.2f}")

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"[INFO] Episode {episode+1}/{max_episodes}, "
                      f"AvgReward(last10): {avg_reward:.2f}, Epsilon={epsilon:.3f}")

            # Save model after every episode
            torch.save(dqn.state_dict(), save_path)
            print(f"[TRAIN] Model saved after Episode {episode+1} to {save_path}")

    # -------------------------------------------------------------------------
    # TEST MODE
    # -------------------------------------------------------------------------
    elif mode == "test":
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model file '{save_path}' not found. Train first.")
        dqn.load_state_dict(torch.load(save_path))
        print(f"[TEST] Loaded model from {save_path}")

        test_model_3min(dqn, rob, action_size, threshold=threshold, run_time=180, success_area=5000)
