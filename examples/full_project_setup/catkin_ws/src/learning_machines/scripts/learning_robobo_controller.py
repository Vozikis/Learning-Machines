#!/usr/bin/env python3
import sys
import json
from robobo_interface import SimulationRobobo, HardwareRobobo
import cv2

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # Initialize variables
    repetitions = 5
    obstacle_detections_per_repetition = 10
    all_sensor_data = []  # To store sensor data across repetitions

    for repetition in range(repetitions):
        print(f"Starting repetition {repetition + 1}")
        obstacle_detections = 0
        sensor_data_per_repetition = []

        while obstacle_detections < obstacle_detections_per_repetition:
            # Read sensor data
            irs = rob.read_irs()
            print("IRS readings:", irs)

            # Define proximity threshold
            threshold = 150
            obstacle_detected = any(value > threshold for value in irs)

            if obstacle_detected:
                print("Obstacle detected! Avoiding_tsouf_tsouf")

                # Save sensor data for this detection
                sensor_data_per_repetition.append(irs)
                obstacle_detections += 1

                # Handle obstacle avoidance
                if any(value > threshold for value in irs[2:6] or irs[7]):
                    rob.move_blocking(-25, -25, 1000)  # Move backward for 1 second
                else:
                    rob.move_blocking(25, 25, 1000)  # Move forward slightly

                if irs[7] >= irs[5]:
                    rob.move_blocking(25, 0, 1000)  # Pivot right for 1 second
                else:
                    rob.move_blocking(0, 25, 1000)

                rob.sleep(0.5)  # Allow time for new sensor readings

            else:
                rob.move_blocking(25, 25, 1000)  # Move forward for 1 second

        # Save sensor data for this repetition
        all_sensor_data.append(sensor_data_per_repetition)
        print(f"Finished repetition {repetition + 1}")



    output_file = "/Users/Antonis/Desktop/sensor_data.json"
    with open(output_file, "w") as f:
        json.dump(all_sensor_data, f, indent=4)
    print(f"Sensor data saved to {output_file}")