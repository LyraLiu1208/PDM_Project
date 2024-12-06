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
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'


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


def run_simulation(debug=False):
    """Run the drone simulation with RRT trajectory planning."""
    # Create the simulation environment
    env = TrajectoryPlanningEnv(drone_model=DEFAULT_DRONES,
                                 num_drones=DEFAULT_NUM_DRONES,
                                 physics=DEFAULT_PHYSICS,
                                 neighbourhood_radius=10,
                                 pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                                 ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                                 gui=DEFAULT_GUI,
                                 record=False,
                                 obstacles=True)

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

    # Initialize PID controllers
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for _ in range(DEFAULT_NUM_DRONES)]

    # Reset environment and start simulation
    # obs = env.reset()
    action = np.zeros((DEFAULT_NUM_DRONES, 4))
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

    env.close()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trajectory Planning with RRT in CtrlAviary')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed output and visualization')
    args = parser.parse_args()

    run_simulation(debug=args.debug)