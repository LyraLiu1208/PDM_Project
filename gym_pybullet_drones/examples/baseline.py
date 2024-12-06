"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
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
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
debug = False

class RRT:
    def __init__(self, start, goal, obstacles, bounds, step_size=0.1, max_iter=1000, debug=False):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]
        self.path = []
        self.debug = debug

    def is_in_collision(self, point):
        """Check if a point collides with any obstacle."""
        for obstacle in self.obstacles:
            center, size = obstacle
            if all(abs(point - center) <= size / 2):
                if self.debug:
                    print(f"Collision detected at {point} with obstacle at {center}")
                return True
        return False

    def get_random_point(self):
        """Generate a random point within bounds."""
        point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        if self.debug:
            print(f"Generated random point: {point}")
        return point

    # def nearest_neighbor(self, point):
    #     """Find the nearest neighbor in the tree to the given point."""
    #     distances = [np.linalg.norm(point - node) for node in self.tree]
    #     return self.tree[np.argmin(distances)]

    def nearest_neighbor(self, point):
        """Find the nearest neighbor in the tree to the given point."""
        distances = []
        for i in range(len(self.tree)):
            distance = [np.linalg.norm(point - self.tree[i])]
            distances.append(distance)
        return self.tree[np.argmin(distances)]
    
    def steer(self, from_node, to_node):
        """Steer from one node towards another by step size."""
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            if self.debug:
                print(f"Direct connection: {from_node} to {to_node}")
            return to_node
        return from_node + (direction / distance) * self.step_size

    def plan(self):
        """Plan a path using RRT."""
        for i in range(self.max_iter):
            rand_point = self.get_random_point()
            nearest = self.nearest_neighbor(rand_point)
            new_point = self.steer(nearest, rand_point)

            if self.debug:
                # Visualize random points and tree connections
                p.addUserDebugLine(nearest, new_point, [0, 1, 0], 1)  # Green for tree edges
                p.addUserDebugText(f"{i}", new_point, textColorRGB=[1, 1, 0], textSize=1.0)  # Yellow for points

            if not self.is_in_collision(new_point):
                self.tree.append(new_point)
                if np.linalg.norm(new_point - self.goal) < self.step_size:
                    self.tree.append(self.goal)
                    self.path = self.construct_path()

                    if self.debug:
                        # Visualize the path
                        for j in range(len(self.path) - 1):
                            p.addUserDebugLine(self.path[j], self.path[j + 1], [0, 0, 1], 2)  # Blue for path
                    print(f"Path found in {i+1} iterations.")
                    return self.path

            # Print progress at every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Tree size = {len(self.tree)}")

        print("Failed to find a path!")
        return []

    def construct_path(self):
        """Construct the path from goal to start."""
        path = [self.goal]
        while True:
            for i, item in enumerate(self.tree):
                if np.array_equal(item, path[-1]):
                    del self.tree[i]
                    break
            nearest = self.nearest_neighbor(path[-1])
            if np.linalg.norm(nearest - self.start) < 1e-2:
                path.append(self.start)
                break
            path.append(nearest)
        if self.debug:
            print(f"Constructed path: {path}")
        return path[::-1]


class TrajectoryPlanningEnv(CtrlAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []
        # self._add_obstacles()

    def _add_obstacles(self):
        """Add static obstacles to the environment."""
        self.obstacles.append((np.array([1, 1, 0.5]), np.array([0.4, 0.4, 0.4])))
        self.obstacles.append((np.array([-1, -1, 0.8]), np.array([0.2, 0.2, 0.6])))
        for center, size in self.obstacles:
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=center)

    def get_obstacles(self):
        """Return the list of obstacles in the environment."""
        return self.obstacles

class TrajectoryPlanningEnv(CtrlAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []
        self._add_obstacles()

    def _add_obstacles(self):
        """Add static obstacles to the environment."""
        self.obstacles.append((np.array([1, 1, 0.5]), np.array([0.4, 0.4, 0.4])))
        self.obstacles.append((np.array([-1, -1, 0.8]), np.array([0.2, 0.2, 0.6])))
        for center, size in self.obstacles:
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=center)

    def get_obstacles(self):
        """Return the list of obstacles in the environment."""
        return self.obstacles

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
        debug = debug
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])


    #### Create the environment ################################
    # env = CtrlAviary(drone_model=drone,
    #                     num_drones=num_drones,
    #                     initial_xyzs=INIT_XYZS,
    #                     initial_rpys=INIT_RPYS,
    #                     physics=physics,
    #                     neighbourhood_radius=10,
    #                     pyb_freq=simulation_freq_hz,
    #                     ctrl_freq=control_freq_hz,
    #                     gui=gui,
    #                     record=record_video,
    #                     obstacles=obstacles,
    #                     user_debug_gui=user_debug_gui
    #                     )

    env = TrajectoryPlanningEnv(drone_model=DEFAULT_DRONES,
                                 num_drones=DEFAULT_NUM_DRONES,
                                 physics=DEFAULT_PHYSICS,
                                 neighbourhood_radius=10,
                                 pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                                 ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                                 gui=DEFAULT_GUI,
                                 record=False,
                                 obstacles=True
                                 )
    
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Begin RRT(*) algorithm here ####

    # Define start, goal, and bounds
    start = np.array([-1, -1, 0.5])
    goal = np.array([1, 1, 0.5])
    bounds = np.array([
        [-2, 2],  # X-axis bounds
        [-2, 2],  # Y-axis bounds
        [0, 1],   # Z-axis bounds
    ])

    # Get obstacles from the environment
    obstacles = env.get_obstacles()

    # Plan path using RRT
    rrt = RRT(start, goal, obstacles, bounds, step_size=0.3, max_iter=5000, debug=debug)
    path = rrt.plan()

    if not path:
        print("RRT failed to find a path!")
        return

    print("RRT path found:", path)
    
    #time.sleep(20)
    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    # if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    obs, reward, terminated, truncated, info = env.step(action)

    # Follow the planned RRT path
    for target in path:
        if debug:
            p.addUserDebugText("Target", target, textColorRGB=[0, 1, 0], textSize=1.2)

        # Move towards each target waypoint
        for _ in range(int(DEFAULT_CONTROL_FREQ_HZ * 0.5)):  # Adjust loop for smooth movement
            for i in range(DEFAULT_NUM_DRONES):
                # Compute control action for the drone
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

        #### Log the simulation ####################################
        # for j in range(num_drones):
        #     logger.log( drone=j,
        #                 timestamp=i / env.CTRL_FREQ,
        #                 state=obs[j],
        #                 control=np.hstack([target, np.zeros(9)])  # Logging the target position with padding for consistency
        #                 )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    # #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("rrt") # Optional CSV save

    # #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
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
