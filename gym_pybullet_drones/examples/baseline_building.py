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
from metrics import Metrics

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.CollapsedBuilding import CollapsedBuildingEnvironment
from collections import defaultdict

NUM_TRIALS = 10
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
debug = False
include_static=True
include_dynamic=False

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
        self.num_iterations = 0
        self.metrics_logger = Metrics()

    def is_in_collision(self, point):
        """Check if a point collides with any obstacle."""
        # for obstacle_id in self.obstacle_ids:
        #     threshold_distance = 0.01
        #     contact_points = p.getClosestPoints(bodyA=obstacle_id, bodyB=-1, distance=threshold_distance)

        #     # Filter for collisions near the point
        #     collision_detected = any(
        #         (abs(cp[5][0] - point[0]) < threshold_distance and
        #         abs(cp[5][1] - point[1]) < threshold_distance and
        #         abs(cp[5][2] - point[2]) < threshold_distance)
        #         for cp in contact_points)
        #     if collision_detected:
        #         return True
        # return False

        for obstacle in self.obstacles:
            center, size = obstacle
            if all(abs(point - center) <= (size+np.array([0.130,0.130,0.030])) / 2):
                # if self.debug:
                #     print(f"Collision detected at {point} with obstacle at {center}")
                return True
        return False

    def edge_in_collision(self, from_node, to_node):
        """Check for collisions along a 3D edge."""
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        steps = int(distance / self.step_size)
        for i in range(1, steps):
            intermediate_point = from_node + (i / steps) * direction
            if self.is_in_collision(intermediate_point):
                return True
        return False

    def get_random_point(self):
        """Generate a random point within bounds."""
        point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        if self.debug:
            print(f"Generated random point: {point}")
        return point
    
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

    def smooth_path(self, path, max_segment_length=1.0):
        smoothed_path = [path[0]]  # Always include the start point
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if not self.edge_in_collision(path[i], path[j]):
                    segment_length = np.linalg.norm(path[i] - path[j])
                    if segment_length <= max_segment_length:
                        break
                j -= 1
            smoothed_path.append(path[j])
            i = j

        smoothed_path.append(path[-1])  # Always include the goal point
        return smoothed_path

    def plan(self):
        """Plan a path using RRT."""
        start_time = time.time()  # Record start time
        for i in range(self.max_iter):
            self.num_iterations += 1 
            rand_point = self.get_random_point()
            nearest = self.nearest_neighbor(rand_point)
            new_point = self.steer(nearest, rand_point)

            if self.debug:
                # Visualize random points and tree connections
                p.addUserDebugLine(nearest, new_point, [0, 1, 0], 1)  # Green for tree edges
                p.addUserDebugText(f"{i}", new_point, textColorRGB=[1, 1, 0], textSize=1.0)  # Yellow for points

            # Check for collisions and add the new point if valid
            if not self.is_in_collision(new_point) and not self.edge_in_collision(nearest, new_point):
                self.tree.append(new_point)
                if np.linalg.norm(new_point - self.goal) < self.step_size:
                    self.tree.append(self.goal)
                    self.path = self.construct_path()
                    if self.debug:
                        # Visualize the path
                        for j in range(len(self.path) - 1):
                            p.addUserDebugLine(self.path[j], self.path[j + 1], [0, 0, 1], 2)  # Blue for path
                    # Apply path smoothing
                    self.path = self.smooth_path(self.path)
                    print(f"Path found in {i+1} iterations.")
                    break

            # Print progress at every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Tree size = {len(self.tree)}")

        end_time = time.time()  # Record end time
        computation_time = end_time - start_time
        path_length = self.metrics_logger.compute_path_length(self.path) if len(self.path) > 0 else None
        path_smoothness = self.metrics_logger.compute_smoothness(self.path) if len(self.path) > 0 else None

        # Store metrics
        metrics = {
            "path_found": len(self.path) > 0,
            "num_iterations": self.num_iterations,
            "computation_time": computation_time,
            "avg_iteration_time": computation_time / self.num_iterations if self.num_iterations > 0 else 0,
            "path_length": path_length,
            "path_smoothness": path_smoothness
        }
        return self.path, metrics

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
    
def run(
        num_trials=NUM_TRIALS,
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
    trial_results = []  # Store results for all trials
    for trial in range(1, num_trials + 1):
        print(f"\n=== Trial {trial} ===")
        #### Initialize the simulation #############################
        # Define start and goal points based on the warehouse layout
        start = np.array([0, -1.0, 0.5])  # Start near the bottom-left corner of the first aisle
        goal = np.array([0, 6.5, 0.5])   # Goal near the top-right corner of the second aisle

        # Define bounds for the warehouse environment
        bounds = np.array([
            [-2, 2],  # X-axis bounds (covering all aisles)
            [-1.5, 8],  # Y-axis bounds
            [0.1, 1.5]    # Z-axis bounds
        ])

        H = .1
        H_STEP = .05
        R = .3
        # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
        INIT_XYZS = np.array([start])
        INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])


        env = CollapsedBuildingEnvironment(drone_model=DEFAULT_DRONES,
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
        obstacles = env.get_obstacles()

        # Plan path using RRT
        print("Planning path using RRT...")
        rrt = RRT(start, goal, obstacles, bounds, step_size=0.3, max_iter=5000, debug=debug)
        path, metrics = rrt.plan()
        
        if path is None or len(path) == 0:
            print("RRT failed to find a path!")
            trial_results.append({
                "trial": trial,
                "path_length": None,
                "path_smoothness": None,
                "num_iterations": metrics["num_iterations"],
                "computation_time": metrics["computation_time"],
                "avg_iteration_time": metrics["avg_iteration_time"]
            })
            env.close()
            continue

        print(f"RRT path found in Trial {trial}:", path)
        trial_results.append({
            "trial": trial,
            "path_length": metrics["path_length"],
            "path_smoothness": metrics["path_smoothness"],
            "num_iterations": metrics["num_iterations"],
            "computation_time": metrics["computation_time"],
            "avg_iteration_time": metrics["avg_iteration_time"]
        })

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

    # Log results
    metrics_logger = Metrics()
    metrics_logger.save_to_yaml(num_trials=num_trials, folder="metrics/Building/RRT", trial_results=trial_results)

    print("\nAll trials completed. Results saved to YAML.")


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
