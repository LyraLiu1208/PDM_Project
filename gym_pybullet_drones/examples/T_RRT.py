import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.WareHouse import WarehouseEnvironment

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_VELOCITY = 0.5
debug = False
include_static=True
include_dynamic=False

class T_RRT:
    def __init__(self, start, goal, static_obstacles, dynamic_obstacles, bounds, velocity=DEFAULT_VELOCITY, step_size=0.1, max_iter=1000, debug=False):
        self.start = np.append(start, 0)  # Add time dimension to the start
        self.goal = np.append(goal, 0)  # Add time dimension to the goal
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.bounds = bounds
        self.velocity = velocity
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]
        self.path = []
        self.debug = debug

    def is_in_collision(self, point, dynamic_obstacles):
        """Check if a point collides with static or dynamic obstacles."""
        for obstacle in self.static_obstacles:
            center, size = obstacle
            if all(abs(point[:3] - center) <= size / 2):
                return True
        for dynamic in dynamic_obstacles:
            obstacle_id, path, index = dynamic
            current_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            current_pos = np.array(current_pos)
            size = path[0][3:]  # Assuming dynamic obstacles include size
            if all(abs(point[:3] - current_pos) <= size / 2):
                return True
        return False

    def steer(self, from_node, to_node):
        """Steer from one node towards another."""
        direction = to_node[:3] - from_node[:3]
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            new_time = from_node[3] + distance / self.velocity
            return np.append(to_node[:3], new_time)
        new_point = from_node[:3] + (direction / distance) * self.step_size
        new_time = from_node[3] + self.step_size / self.velocity
        return np.append(new_point, new_time)

    def get_random_point(self):
        """Generate a random point within bounds."""
        position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return np.append(position, 0)  # Add time dimension

    def nearest_neighbor(self, point):
        """Find the nearest neighbor in the tree to the given point."""
        distances = [np.linalg.norm(node[:3] - point[:3]) for node in self.tree]
        return self.tree[np.argmin(distances)]

    def plan(self, env):
        """Plan a path using T-RRT."""
        for i in range(self.max_iter):
            env.update_dynamic_obstacles()  # Update positions of dynamic obstacles
            dynamic_obstacles = env.get_obstacles()[1]  # Get updated dynamic obstacles

            rand_point = self.get_random_point()
            nearest = self.nearest_neighbor(rand_point)
            new_point = self.steer(nearest, rand_point)

            if not self.is_in_collision(new_point, dynamic_obstacles):
                self.tree.append(new_point)
                if np.linalg.norm(new_point[:3] - self.goal[:3]) < self.step_size:
                    self.tree.append(self.goal)
                    self.path = self.construct_path()
                    return self.path
        print("Failed to find a path!")
        return []

    def construct_path(self):
        """Construct the path from goal to start."""
        path = [self.goal]
        while True:
            nearest = self.nearest_neighbor(path[-1])
            path.append(nearest)
            if np.array_equal(nearest[:3], self.start[:3]):
                break
        return path[::-1]

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        debug=debug
):
    """
    Execute the RRT path planning in the warehouse environment.
    """
    # Define start and goal points based on the warehouse layout
    start = np.array([0.5, -1.0, 0.3])  # Start near the bottom-left corner of the first aisle
    goal = np.array([3.5, 5.0, 0.1])   # Goal near the top-right corner of the second aisle

    # Define bounds for the warehouse environment
    bounds = np.array([
        [-0.5, 4.5],  # X-axis bounds (covering all aisles)
        [-1.5, 9.0],  # Y-axis bounds
        [0.0, 1.0]    # Z-axis bounds
    ])

    # Initialize the warehouse environment
    H = .1
    H_STEP = .05
    R = .3
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_XYZS = np.array([start])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])

    env = WarehouseEnvironment( include_static=include_static, 
                                include_dynamic=include_dynamic,
                                drone_model=DEFAULT_DRONES,
                                num_drones=DEFAULT_NUM_DRONES,
                                initial_xyzs=INIT_XYZS,
                                initial_rpys=INIT_RPYS,
                                physics=DEFAULT_PHYSICS,
                                neighbourhood_radius=10,
                                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                                gui=DEFAULT_GUI,
                                record=False,
                                obstacles=False
                                 )
    PYB_CLIENT = env.getPyBulletClient()

    # Get static and dynamic obstacles
    static_obstacles, dynamic_obstacles = env.get_obstacles()

    # Plan path using T-RRT
    print("Planning path using T-RRT...")
    trrt = T_RRT(start, goal, static_obstacles, dynamic_obstacles, bounds, step_size=0.3, max_iter=5000, debug=debug)
    path = trrt.plan(env)

    if not path:
        print("T-RRT failed to find a path.")
        return

    print("T-RRT path found:", path)

    #### Initialize the controllers ############################
    # if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    k = 0
    # Follow the planned RRT path
    for target in path:
        k = k + 1
        if debug:
            p.addUserDebugText("Target", target, textColorRGB=[0, 1, 0], textSize=1.2)

        # Move towards each target waypoint
        for step in range(0, int(2*env.CTRL_FREQ)):  # Adjust loop for smooth movement
            for i in range(DEFAULT_NUM_DRONES):
                # Compute control action for the drone
                if include_dynamic:
                    env.update_dynamic_obstacles()
                action[i, :], _, _ = ctrl[i].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[i],
                    target_pos=target,
                    target_rpy=np.zeros(3)  # Assuming level flight
                )

            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            if debug:
                env.render()
            if terminated or truncated:
                print("Simulation terminated early.")
                break
            
            env.render()
            sync(step + (k-1)*int(2*env.CTRL_FREQ), START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))