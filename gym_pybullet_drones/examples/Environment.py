import os
import time
import argparse
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.WareHouse import WarehouseEnvironment  # Updated import
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
debug = False

def run():
    """
    Initialize and render the warehouse environment without any planning.
    """
    env = WarehouseEnvironment(include_static=True, include_dynamic=True, drone_model=DEFAULT_DRONES, gui=DEFAULT_GUI)
    PYB_CLIENT = env.getPyBulletClient()

    # Render the environment to visualize obstacles and dynamics
    print("Rendering the warehouse environment...")

    try:
        # Keep the simulation running to observe the environment
        for _ in range(1000):  # Render for 1000 timesteps
            env.update_dynamic_obstacles()  # Update positions of dynamic obstacles (workers)
            env.render()  # Render the environment
            time.sleep(1 / DEFAULT_SIMULATION_FREQ_HZ)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    # Close the environment
    env.close()

if __name__ == "__main__":
    run()