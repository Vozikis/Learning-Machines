#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
# from learning_machines.test_actions import *
import cv2

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


# def test_emotions(rob: IRobobo):
#     rob.set_emotion(Emotion.HAPPY)
#     rob.talk("Hello")
#     rob.play_emotion_sound(SoundEmotion.PURR)
#     rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


# def test_move_and_wheel_reset(rob: IRobobo):
#     rob.move_blocking(100, 100, 1000) 
#     print("before reset: ", rob.read_wheels())
#     print("test")
#     rob.reset_wheels()
#     rob.sleep(1)
#     print("after reset: ", rob.read_wheels())
    
    
# def test_move_and_wheel_reset2(rob: IRobobo):
#     rob.move_blocking(0, 100, 1000) 
#     print("before reset: ", rob.read_wheels())
#     print("test")
#     rob.reset_wheels()
#     rob.sleep(1)
#     print("after reset: ", rob.read_wheels())


# def test_sensors(rob: IRobobo):
#     print("IRS data: ", rob.read_irs())
#     image = rob.read_image_front()
#     cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
#     print("Phone pan: ", rob.read_phone_pan())
#     print("Phone tilt: ", rob.read_phone_tilt())
#     print("Current acceleration: ", rob.read_accel())
#     print("Current orientation: ", rob.read_orientation())


# def test_phone_movement(rob: IRobobo):
#     rob.set_phone_pan_blocking(20, 100)
#     print("Phone pan after move to 20: ", rob.read_phone_pan())
#     rob.set_phone_tilt_blocking(50, 100)
#     print("Phone tilt after move to 50: ", rob.read_phone_tilt())


# def test_sim(rob: SimulationRobobo):
#     print("Current simulation time:", rob.get_sim_time())
#     print("Is the simulation currently running? ", rob.is_running())
#     rob.stop_simulation()
#     print("Simulation time after stopping:", rob.get_sim_time())
#     print("Is the simulation running after shutting down? ", rob.is_running())
#     rob.play_simulation()
#     print("Simulation time after starting again: ", rob.get_sim_time())
#     print("Current robot position: ", rob.get_position())
#     print("Current robot orientation: ", rob.get_orientation())

#     pos = rob.get_position()
#     orient = rob.get_orientation()
#     rob.set_position(pos, orient)
#     print("Position the same after setting to itself: ", pos == rob.get_position())
#     print("Orient the same after setting to itself: ", orient == rob.get_orientation())


# def test_hardware(rob: HardwareRobobo):
#     print("Phone battery level: ", rob.phone_battery())
#     print("Robot battery level: ", rob.robot_battery())


# def run_all_actions(rob: IRobobo):
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
#     test_emotions(rob)
#     test_sensors(rob)
#     test_move_and_wheel_reset(rob)
#     if isinstance(rob, SimulationRobobo):
#         test_sim(rob)

#     if isinstance(rob, HardwareRobobo):
#         test_hardware(rob)

#     test_phone_movement(rob)

#     if isinstance(rob, SimulationRobobo):
#         rob.stop_simulation()









# if __name__ == "__main__":
#     # You can do better argument parsing than this!
#     if len(sys.argv) < 2:
#         raise ValueError(
#             """To run, we need to know if we are running on hardware of simulation
#             Pass `--hardware` or `--simulation` to specify."""
#         )
#     elif sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=True)
#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#     else:
#         raise ValueError(f"{sys.argv[1]} is not a valid argument.")
#     while(1):
#         test_sensors(rob)
#         test_move_and_wheel_reset(rob)
#         test_sensors(rob)
#         k = rob.read_irs()
#         test_move_and_wheel_reset(rob)
#         test_sensors(rob)
#         l = rob.read_irs()
#         t = []
#         for i in range(len(k)):
#             t.append( k[i] - l[i])
#             if not(t[i] < 0 and t[i] > -2 or t[i] > 0 and t[i] < 2): 
#                 rob.move_blocking(0, 100, 1000)

#     # t = k-l
#     # for i in range(len(t)):
#     #     print(t[i])
#     # run_all_actions(rob)

#!/usr/bin/env python3
#!/usr/bin/env python3
# import sys
# from robobo_interface import SimulationRobobo, HardwareRobobo

# def main():
#     if len(sys.argv) < 2:
#         raise ValueError(
#             """To run, we need to know if we are running on hardware or simulation.
#             Pass `--hardware` or `--simulation` to specify."""
#         )
#     elif sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=True)
#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#     else:
#         raise ValueError(f"{sys.argv[1]} is not a valid argument.")

#     try:
#         if isinstance(rob, SimulationRobobo):
#             rob.play_simulation()

#         prev_irs_data = rob.read_irs()  # Initialize with the first sensor reading

#         while True:
#             irs_data = rob.read_irs()
#             print("Current IR sensor data:", irs_data)
#             print("Previous IR sensor data:", prev_irs_data)

#             # Compute the difference between current and previous IR readings
#             differences = [current - previous for current, previous in zip(irs_data, prev_irs_data)]
#             print("Differences in IR sensor data:", differences)

#             # Check if any difference exceeds a threshold (indicating a potential obstacle)
#             if any(abs(diff) > 5 for diff in differences):  # Threshold for detecting changes
#                 print("Significant sensor change detected! Changing direction.")
#                 rob.move_blocking(-100, -100, 500)  # Move backward briefly
#                 rob.move_blocking(100, -100, 1000)  # Turn in place
#             else:
#                 # Move forward if no significant change is detected
#                 rob.move_blocking(100, 100, 1000)

#             # Update the previous IR sensor data
#             prev_irs_data = irs_data

#     except KeyboardInterrupt:
#         print("Exiting program.")
#     finally:
#         if isinstance(rob, SimulationRobobo):
#             rob.stop_simulation()

# if __name__ == "__main__":
#     main()

#!/usr/#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import random
from robobo_interface import SimulationRobobo, HardwareRobobo

def calculate_moving_average(data, window_size=5):
    """Calculate a simple moving average for smoother data."""
    return [sum(data[max(0, i - window_size + 1):i + 1]) / (i - max(0, i - window_size + 1) + 1) for i in range(len(data))]

def main():
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation.
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    try:
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()

        recovery_mode = False 
        recovery_steps = 5  
        average_threshold = 60 

        while True:
            irs_data = rob.read_irs()
            print("Raw IR sensor data:", irs_data)

            average_irs = sum(irs_data) / len(irs_data)
            print(f"Average IR sensor value: {average_irs}")

            if average_irs > average_threshold and not recovery_mode:
                print("Object detected! Initiating recovery behavior.")
                rob.move_blocking(-200, -200, 1000) 

                random_angle = random.randint(120, 180)
                random_direction = random.choice([-1, 1])
                rob.move_blocking(100 * random_direction, -100 * random_direction, random_angle * 10)

                recovery_mode = True  
                recovery_steps = 5 

            if recovery_mode:
                recovery_steps -= 1
                if recovery_steps <= 0:
                    recovery_mode = False  
            if not recovery_mode:
                print("No obstacles detected. Moving forward.")
                rob.move_blocking(100, 100, 1000)

    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

if __name__ == "__main__":
    main()
