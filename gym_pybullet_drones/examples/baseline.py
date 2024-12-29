import time
import argparse
import numpy as np
import pybullet as p
from scipy.spatial import KDTree
from metrics import Metrics

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
debug = False
include_static=True
include_dynamic=False

class RRT:
    def __init__(self, start, goal, obstacles, bounds, step_size=0.1, max_iter=1000, debug=False):
        """
        Initialize the RRT planner.

        Args:
            start (list or np.array): Starting point in 3D space [x, y, z].
            goal (list or np.array): Goal point in 3D space [x, y, z].
            obstacles (list): List of obstacles, each represented as a tuple (center, size).
                - center (np.array): The center position of the obstacle [x, y, z].
                - size (np.array): The size (width, height, depth) of the obstacle.
            bounds (np.array): Spatial bounds for random sampling, shaped as [[min_x, max_x], [min_y, max_y], [min_z, max_z]].
            step_size (float, optional): Maximum step size for steering in each iteration. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the planning algorithm. Defaults to 1000.
            debug (bool, optional): Enable debug messages and visualizations. Defaults to False.

        Attributes:
            tree (list): List of nodes in the RRT tree.
            path (list): List of nodes forming the planned path.
            parents (dict): Dictionary to track parent-child relationships for tree nodes.
            debug (bool): Enable or disable debug logs.
        Raises:
            ValueError: If inputs are invalid or incorrectly formatted.
        """
        # Validate start and goal positions
        if len(start) != 3 or len(goal) != 3:
            raise ValueError("Start and goal positions must be 3D points [x, y, z].")

        # Validate bounds
        if bounds.shape != (3, 2):
            raise ValueError("Bounds must be a 3x2 array defining [min, max] for each axis (x, y, z).")
        if not np.all(bounds[:, 0] <= bounds[:, 1]):
            raise ValueError("Invalid bounds: min values must be less than or equal to max values.")

        # Validate obstacles
        for obs in obstacles:
            if len(obs) != 2 or len(obs[0]) != 3 or len(obs[1]) != 3:
                raise ValueError("Each obstacle must be a tuple (center, size), where both are 3D vectors.")

        # Initialize RRT attributes
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles  # Obstacles are represented as (center, size) tuples
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter

        self.tree = [self.start]
        self.path = []
        self.parents = {tuple(self.start): None}  # Start node has no parent
        self.debug = debug
        self.metrics = Metrics()

    def is_in_collision(self, point):
        """
        Check if a point is in collision with any obstacle.

        Args:
            point (np.array): A 3D point in the configuration space.
            padding (np.array): Padding to add around obstacles for collision checks.

        Returns:
            bool: True if the point is in collision, False otherwise.
        """
        padding=np.array([0.1, 0.1, 0.1])
    
        for obstacle in self.obstacles:
            center, size = obstacle
            half_size = size / 2

            # Add padding to the obstacle bounds
            lower_bound = center - half_size - padding
            upper_bound = center + half_size + padding

            # Check if the point lies within the padded obstacle bounds
            if np.all(lower_bound <= point) and np.all(point <= upper_bound):
                if self.debug:
                    print(f"Collision detected: Point {point} intersects obstacle at {center}")
                return True

        if self.debug:
            print(f"No collision: Point {point} is collision-free.")

        return False

    def edge_in_collision(self, from_node, to_node):
        """
        Check if an edge between two nodes is in collision with any obstacle.

        Args:
            from_node (np.array): Start node of the edge.
            to_node (np.array): End node of the edge.

        Returns:
            bool: True if the edge is in collision, False otherwise.
        """
        # Calculate the direction and distance between nodes
        direction = to_node - from_node
        distance = np.linalg.norm(direction)

        # Dynamically determine the number of steps for collision checks
        steps = max(1, int(distance / self.step_size))
        step_vector = direction / steps

        # Iterate through intermediate points along the edge
        for step in range(1, steps + 1):
            intermediate_point = from_node + step * step_vector

            # Check if the intermediate point is in collision
            if self.is_in_collision(intermediate_point):
                if self.debug:
                    print(f"Collision detected on edge: {from_node} -> {to_node}, at {intermediate_point}")
                return True

        return False

    def get_random_point(self, max_retries=100):
        """
        Generate a random collision-free point within bounds.

        Args:
            max_retries (int, optional): Maximum number of attempts to find a collision-free point. Defaults to 100.

        Returns:
            np.array: A collision-free point within bounds.

        Raises:
            RuntimeError: If a valid point cannot be found within the retry limit.
        """
        for attempt in range(max_retries):
            # Uniform random sampling within bounds
            point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            # Debugging: Log the sampled point
            if self.debug:
                print(f"Sampled random point: {point}")

            # Check if the sampled point is collision-free
            if not self.is_in_collision(point):
                return point

        # Raise error if no valid point is found after max retries
        raise RuntimeError("Failed to generate a collision-free point after maximum retries.")
    
    def nearest_neighbor(self, point):
        """
        Find the nearest neighbor in the tree to a given point.

        Args:
            point (np.array): The target point for which the nearest neighbor is sought.

        Returns:
            np.array: The nearest neighbor node in the tree.
        """
        # Handle edge cases for empty or malformed trees
        if not self.tree:
            raise ValueError("The tree is empty. Cannot find a nearest neighbor.")

        # Ensure KDTree is up-to-date
        if not hasattr(self, 'kd_tree') or len(self.tree) != self.kd_tree.n:
            self.kd_tree = KDTree(self.tree)

        # Query the KDTree for the nearest neighbor
        distance, index = self.kd_tree.query(point)
        nearest_node = self.tree[index]

        # Debugging: Log nearest neighbor details
        if self.debug:
            print(f"Nearest neighbor to {point}: {nearest_node} at distance {distance}")

        return nearest_node
    
    def steer(self, from_node, to_node, step_size=None):
        """
        Steer from a starting node toward a target node by a specified step size.

        Args:
            from_node (np.array): Starting node [x, y, z].
            to_node (np.array): Target node [x, y, z].
            step_size (float, optional): Maximum step size. Defaults to `self.step_size`.

        Returns:
            np.array: A new node in the direction of `to_node`, within the step size.
        """
        if step_size is None:
            step_size = self.step_size

        # Compute the direction vector from `from_node` to `to_node`
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)

        # Handle edge cases where nodes are very close or identical
        if distance < 1e-6:
            if self.debug:
                print(f"Nodes are too close or identical: {from_node} -> {to_node}")
            return np.array(from_node)

        # Normalize the direction vector
        unit_direction = direction / distance

        # Compute the new node, ensuring it does not overshoot the target
        step_distance = min(step_size, distance)
        new_node = np.array(from_node) + step_distance * unit_direction

        # Clamp the new node to stay within the bounds
        new_node = np.clip(new_node, self.bounds[:, 0], self.bounds[:, 1])

        # Debugging: Log the steering details
        if self.debug:
            print(f"Steering from {from_node} toward {to_node} with step size {step_distance}")
            print(f"New node: {new_node}")

        return new_node

    def smooth_path(self, path, max_segment_length=1.0):
        """
        Smooth the path by iteratively removing unnecessary waypoints while ensuring collision-free segments.

        Args:
            path (list): The original path as a list of nodes.
            max_segment_length (float): Maximum allowable segment length after smoothing.

        Returns:
            list: A smoothed path with reduced waypoints.
        """
        if len(path) <= 2:
            return path  # Path is already smooth

        smoothed_path = [path[0]]  # Always include the start point
        i = 0

        while i < len(path) - 1:
            # Find the furthest node that can form a collision-free segment with the current node
            for j in range(len(path) - 1, i, -1):
                if not self.edge_in_collision(path[i], path[j]):
                    segment_length = np.linalg.norm(path[i] - path[j])
                    if segment_length <= max_segment_length:
                        smoothed_path.append(path[j])
                        i = j
                        break
            else:
                # If no valid segment is found, move to the next point
                smoothed_path.append(path[i + 1])
                i += 1

        # Ensure the goal point is included
        if not np.array_equal(smoothed_path[-1], path[-1]):
            smoothed_path.append(path[-1])

        # Debugging: Log the smoothed path
        if self.debug:
            print(f"Original path: {path}")
            print(f"Smoothed path: {smoothed_path}")

        return smoothed_path

    def plan(self, goal_bias=0.1):
        """
        Plan a path from the start to the goal using the RRT algorithm.

        Args:
            goal_bias (float): Probability of sampling the goal point directly. Defaults to 0.1.

        Returns:
            list: A smoothed path from start to goal if found, otherwise an empty list.

        Raises:
            RuntimeError: If no valid path is found within the maximum iterations.
        """
        start_time = time.time()  # Start timing the planning process

        for iteration in range(self.max_iter):
            # Sample a random point with goal biasing
            if np.random.random() < goal_bias:
                random_point = self.goal
            else:
                random_point = self.get_random_point()

            # Find the nearest node to the sampled point
            nearest_node = self.nearest_neighbor(random_point)

            # Steer towards the sampled point
            new_node = self.steer(nearest_node, random_point)

            # Check if the new node is collision-free
            if not self.is_in_collision(new_node) and not self.edge_in_collision(nearest_node, new_node):
                # Add the new node to the tree
                self.tree.append(new_node)

                # Check if the goal is reached
                if np.linalg.norm(new_node - self.goal) < self.step_size:
                    self.tree.append(self.goal)
                    break

            # Log progress at regular intervals
            if self.debug and iteration % 100 == 0:
                print(f"Iteration {iteration}: Tree size = {len(self.tree)}")

        # Check if the goal node is connected to the tree
        goal_reached = any(np.array_equal(self.goal, node) for node in self.tree)
        if not goal_reached:
            print("Failed to find a path!")
            return []

        # Construct the final path
        path = self.construct_path()
        smoothed_path = self.smooth_path(path)

        # Collect metrics for evaluation
        self.metrics.compute_computation_time(start_time, time.time())
        self.metrics.compute_path_length(smoothed_path)
        self.metrics.compute_smoothness(smoothed_path)

        # Debugging: Log final metrics
        if self.debug:
            print(f"Path found: {smoothed_path}")
            print(f"Path length: {self.metrics.data['path_length']}")
            print(f"Path smoothness: {self.metrics.data['smoothness']}")
            print(f"Computation time: {self.metrics.data['computation_time']} seconds")

        return smoothed_path

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
    obstacles, _ = env.get_obstacles()

    # Plan path using RRT
    print("Planning path using RRT...")
    rrt = RRT(start, goal, obstacles, bounds, step_size=0.3, max_iter=5000, debug=debug)
    path = rrt.plan()

    # Collect metrics for this trial
    trial_results = [{
        "trial": 1,
        "path_length": rrt.metrics.data.get("path_length", 0),
        "smoothness": rrt.metrics.data.get("smoothness", 0),
        "computation_time": rrt.metrics.data.get("computation_time", 0)
    }]

    # Save metrics to a YAML file
    rrt.metrics.save_to_yaml(folder="metrics/WareHouse/RRT", trial_results=trial_results)
    print("Metrics saved successfully.")

    p.removeAllUserDebugItems()
    if path is not None:
        for j in range(len(path) - 1):
                p.addUserDebugLine(path[j], path[j + 1], [0, 0.6, 0], 2)

    if not path:
        print("RRT failed to find a path!")
        return

    print("RRT path found:", path)

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
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool, help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool, help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))