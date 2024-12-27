import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy.spatial import KDTree

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

class RRT_STAR:
    def __init__(self, start, goal, obstacles, obstacle_ids, bounds, step_size=0.1, max_iter=1000, debug=False):
        """
        Initialize the RRT* planner.

        Args:
            start (np.array): Start position [x, y, z].
            goal (np.array): Goal position [x, y, z].
            obstacles (list): List of obstacles, where each obstacle is represented as a tuple (center, size).
                - center (np.array): The center position of the obstacle [x, y, z].
                - size (np.array): The size (width, height, depth) of the obstacle.
            obstacle_ids (list): List of obstacle IDs (optional, used for PyBullet integration).
            bounds (np.array): Spatial bounds for random sampling, shaped as [[min_x, max_x], [min_y, max_y], [min_z, max_z]].
            step_size (float): Maximum step size for steering in each iteration.
            max_iter (int): Maximum number of iterations for the planning algorithm.
            debug (bool): Enable debug messages and visualizations.

        Attributes:
            tree (list): List of nodes in the RRT* tree.
            path (list): List of nodes forming the planned path.
            obstacle_centers (np.array): Extracted centers of all obstacles for KD-Tree construction.
            obstacle_kd_tree (scipy.spatial.KDTree): Static KDTree for obstacle centers.
            node_kd_tree (scipy.spatial.KDTree): Dynamic KDTree for RRT* tree nodes.
            costs (dict): Dictionary storing the cost to reach each node.
            parents (dict): Dictionary storing parent-child relationships for tree nodes.
            children (defaultdict): Dictionary mapping nodes to their children.
            radius_const (float): Constant used to calculate neighbor search radius.
            edges (dict): Dictionary storing edges between nodes in the tree for connection tracking.
        
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
    
        # Initialize RRT* planner attributes
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles  # Obstacles are represented as (center, size) tuples
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter

        self.tree = [np.array(start)]
        self.path = []
        self.debug = debug
        self.obstacle_ids = obstacle_ids
        
        # Initialize KDTree for obstacle centers for efficient nearest neighbor queries
        self.obstacle_centers = np.array([obs[0] for obs in obstacles])  # Extract obstacle centers
        
        # Build KDTree for fast spatial indexing 
        self.obstacle_kd_tree = KDTree(self.obstacle_centers)
        self.node_kd_tree = None 

        self.costs = {tuple(self.start): 0}  # Start node cost is 0
        self.parents = {tuple(self.start): None}  # Start node has no parent
        self.children = defaultdict(list)  # Tracks children of each node
        self.radius_const = 3  # Used for neighbor search radius
        self.edges = {}  # Dictionary to store edges between nodes

    def is_in_collision(self, point):
        """
        Check if a point is in collision with any obstacle using KD-Tree for optimization.

        Why it's needed:
        - Reduces computational overhead by limiting collision checks to nearby obstacles.
        - Ensures fast queries in environments with many obstacles.

        Args:
            point (np.array): A 3D point in the configuration space.

        Returns:
            bool: True if the point is in collision with any obstacle, False otherwise.
        """
        collision_threshold = 1.0
        nearby_indices = self.obstacle_kd_tree.query_ball_point(point, collision_threshold)

        for idx in nearby_indices:
            center, size = self.obstacles[idx]
            half_size = size / 2
            padding = np.array([0.130, 0.130, 0.030])
            lower_bound = center - half_size - padding
            upper_bound = center + half_size + padding 

            # Check if the point lies within the bounding box
            if np.all(lower_bound <= point) and np.all(point <= upper_bound):
                if self.debug:
                    print(f"Collision detected: Point {point} intersects obstacle at {center}")
                return True 
            
        return False

    def get_random_point(self, max_retries=100):
        """
        Generate a random collision-free point within the defined bounds.

        Why it's needed:
            - Provides random points for tree expansion while avoiding collisions.

        Args:
            max_retries (int): Maximum number of attempts to find a collision-free point.

        Returns:
            np.array: A collision-free point within bounds.

        Raises:
            RuntimeError: If a valid point cannot be found within the retry limit.
        """
        for _ in range(max_retries):
            # Uniform random sampling within bounds
            point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            # Check if the point is collision-free
            if not self.is_in_collision(point):
                if self.debug:
                    print(f"Generated collision-free point: {point}")
                return point

        # If no valid point is found after max retries, raise an error
        raise RuntimeError("Failed to generate a collision-free point after maximum retries.")

    def nearest_neighbor(self, point):
        """
        Find the nearest neighbor in the tree to a given point, considering static obstacles.

        Why it's needed:
            - Guarantees safe tree expansion by ensuring selected neighbors are valid.

        Args:
            point (np.array): A 3D point in the configuration space.

        Returns:
            np.array: The nearest neighbor node in the tree.
        """
        # Ensure KDTree for nodes is up-to-date
        if not self.node_kd_tree or len(self.tree) != self.node_kd_tree.n:
            self.node_kd_tree = KDTree(self.tree)

        # Query the KDTree for the nearest node
        distance, index = self.node_kd_tree.query(point)

        return self.tree[index]
    
    def steer(self, from_node, to_node, step_size=None):
        """
        Steer from a starting node toward a target node by a specified step size.

        Why it's needed:
        - Expands the RRT* tree incrementally toward the target node.

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

        # Handle edge cases where nodes are extremely close
        if distance < 1e-6:
            if self.debug:
                print(f"Nodes are too close or identical: {from_node} -> {to_node}")
            return np.array(from_node)

        # Normalize the direction vector
        unit_direction = direction / distance

        # Compute the new node, ensuring it does not overshoot the target
        step_distance = min(step_size, distance)
        new_node = np.array(from_node) + step_distance * unit_direction

        # Debugging information
        if self.debug:
            print(f"Steering from {from_node} toward {to_node} with step size {step_distance}")
            print(f"New node: {new_node}")

        return new_node

    def near(self, node, radius=None):
        """
        Find all nodes within a specified radius of a given node.

        Args:
            node (np.array): The node for which neighbors are sought.
            radius (float, optional): The search radius. Defaults to a calculated radius.

        Returns:
            list: A list of nodes within the specified radius.
        """
        if radius is None:
            radius = self.radius_const * (np.log(len(self.tree)) / len(self.tree)) ** (1 / 3)

        # Ensure KDTree for nodes is up-to-date
        if not self.node_kd_tree or len(self.tree) != self.node_kd_tree.n:
            self.node_kd_tree = KDTree(self.tree)

        # Query the KDTree for neighbors
        indices = self.node_kd_tree.query_ball_point(node, radius)

        return [self.tree[idx] for idx in indices]

    def cost(self, node):
        """
        Compute the cost of reaching a node from the start.

        Why it's needed:
            - Determines the total path cost to a node, which is critical for path optimization and rewiring.

        Args:
            node (np.array): The node for which the cost is calculated.

        Returns:
            float: The total cost to reach the node from the start.

        Raises:
            ValueError: If the node is not in the tree or does not have a valid parent.
        """
        if tuple(node) not in self.costs:
            raise ValueError(f"Node {node} is not in the tree or does not have a valid cost entry.")

        total_cost = 0.0
        current_node = node

        # Traverse the parent chain to sum up costs
        while current_node is not None:
            parent_node = self.parents.get(tuple(current_node))
            if parent_node is None:  # Start node reached
                break
            edge_cost = np.linalg.norm(np.array(current_node) - np.array(parent_node))
            total_cost += edge_cost
            current_node = parent_node

        if self.debug:
            print(f"Total cost to reach node {node}: {total_cost}")

        return total_cost

    def rewire(self, new_node):
        """ Rewires path to new node if better one exists"""
        for neighbor in self.near(new_node):
            if not self.edge_in_collision(neighbor, new_node):
                new_cost = self.cost(new_node) + np.linalg.norm(new_node - neighbor)
                if new_cost < self.cost(neighbor):
                    self.alter_parent(neighbor, new_node)
                    #self.parents[tuple(neighbor)] = new_node
                    self.costs[tuple(neighbor)] = new_cost
                    self.propagate_cost_updates(neighbor)
                
    def propagate_cost_updates(self, node):
        for child in self.children_of(node):
            self.costs[tuple(child)] = self.costs[tuple(node)] + np.linalg.norm(child - node)
            # Recursively update its children
            self.propagate_cost_updates(child)

    def children_of(self, node):
        """Return the list of children of a given node."""
        return self.children.get(tuple(node), [])

    def lowest_cost_neighbor(self, new_node):
        best_neighbor = []
        costs = []
        neighbors = []
        found = False
        #for i in range(len(self.tree)):
        for neighbor in self.near(new_node):
            #neighbor = self.tree[i]
            neighbors.append(neighbor)
            costs.append(self.cost(neighbor) + np.linalg.norm(new_node - neighbor))
        indices = np.argsort(costs)
        for index in indices:
            if not self.edge_in_collision(neighbors[int(index)], new_node):
                self.costs[tuple(new_node)] = costs[int(index)]
                best_neighbor = neighbors[int(index)]
                found = True
                break
            
        if not found:
            return self.nearest_neighbor(new_node)
        else:
            return best_neighbor

    def edge_in_collision(self, from_node, to_node):
        direction = to_node - from_node
        nr = 50
        for i in range(1,nr-1):
            sample = from_node + 1/nr*(i)*direction
            if self.is_in_collision(sample):
                return True
        return False

    def alter_parent(self, node, new_parent):
        # if self.parents.get(tuple(node)) is None:
        #     self.parents[tuple(node)] = new_parent
        #     self.edges[(tuple(node), tuple(new_parent))] = p.addUserDebugLine(node, self.parents[tuple(node)], [1, 0, 0], 3)
        # else:
        #     p.removeUserDebugItem(self.edges[(tuple(node),tuple(self.parents[tuple(node)]))])
        #     self.parents[tuple(node)] = new_parent
        #     self.edges[(tuple(node), tuple(new_parent))] = p.addUserDebugLine(node, self.parents[tuple(node)], [1, 0, 0], 3)
        #     self.children[tuple(self.parents.get(tuple(node)))].remove(node)
        # self.children[tuple(new_parent)].append(node)
        """Change the parent of a node and update edges and children tracking."""
        node_tuple = tuple(node)
        new_parent_tuple = tuple(new_parent)

        # Remove edge and update children tracking for the old parent
        old_parent = self.parents.get(node_tuple)
        if old_parent is not None:
            # Remove the visual debug line for the old edge
            if (node_tuple, tuple(old_parent)) in self.edges:
                #p.removeUserDebugItem(self.edges[(node_tuple, tuple(old_parent))])
                del self.edges[(node_tuple, tuple(old_parent))]
            # Remove this node from the old parent's children list
            if tuple(old_parent) in self.children:
                self.children[tuple(old_parent)] = [
                    child for child in self.children[tuple(old_parent)] if not np.array_equal(child, node)
                ]

        # Set the new parent
        self.parents[node_tuple] = new_parent

        # Add the visual debug line for the new edge
        #self.edges[(node_tuple, new_parent_tuple)] = p.addUserDebugLine(node, new_parent, [1, 0, 0], 3)

        # Add this node as a child of the new parent
        if new_parent_tuple not in self.children:
            self.children[new_parent_tuple] = []
        self.children[new_parent_tuple].append(node)
    
    def visualize_tree_bullet(self):
        """
        Visualize the branching structure of the RRT* tree in the Bullet physics example browser.

        Draws the entire tree at the end of the planning process.
        """
        # Draw all nodes as small blue spheres
        for node in self.tree:
            p.addUserDebugText(
                text="o",  # Render a small sphere (symbolic with 'o')
                textPosition=node,
                textColorRGB=[0, 0, 1],  # Blue for nodes
                textSize=1.2,
                lifeTime=0  # Persist until explicitly removed
            )

        # Draw edges (branches) connecting nodes to their parents
        for node in self.tree:
            parent = self.parents.get(tuple(node))
            if parent is not None:
                p.addUserDebugLine(
                    lineFromXYZ=parent,
                    lineToXYZ=node,
                    lineColorRGB=[0.6, 0.6, 0.6],  # Gray for edges
                    lineWidth=3.0,  # Thicker lines for better visibility
                    lifeTime=0  # Persist until explicitly removed
                )

        # Highlight start and goal nodes
        p.addUserDebugText(
            text="START",
            textPosition=self.start,
            textColorRGB=[0, 1, 0],  # Green for start
            textSize=1.5,
            lifeTime=0
        )
        p.addUserDebugText(
            text="GOAL",
            textPosition=self.goal,
            textColorRGB=[1, 0, 0],  # Red for goal
            textSize=1.5,
            lifeTime=0
        )

    def visualize_path_bullet(self, path):
        """
        Visualize the optimal path in the Bullet physics example browser.

        Args:
            path (list): List of nodes forming the optimal path.
        """
        for i in range(len(path) - 1):
            p.addUserDebugLine(
                lineFromXYZ=path[i],
                lineToXYZ=path[i + 1],
                lineColorRGB=[1, 0.5, 0],  # Orange for optimal path
                lineWidth=5.0,  # Thicker lines for the path
                lifeTime=0  # Persist until explicitly removed
            )

    def plan(self):
        """
        Plan a path from the start to the goal using the RRT* algorithm.
        """
        self.debug_items = []  # Initialize the list to track debug items

        for iteration in range(self.max_iter):
            random_point = self.get_random_point()
            nearest_node = self.nearest_neighbor(random_point)
            new_node = self.steer(nearest_node, random_point)

            if not self.is_in_collision(new_node):
                self.tree.append(new_node)
                self.parents[tuple(new_node)] = tuple(nearest_node)
                self.costs[tuple(new_node)] = self.costs[tuple(nearest_node)] + np.linalg.norm(
                    np.array(new_node) - np.array(nearest_node)
                )

                # Optional rewiring logic
                self.rewire(new_node)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < self.step_size:
                    self.parents[tuple(self.goal)] = tuple(new_node)
                    self.costs[tuple(self.goal)] = self.costs[tuple(new_node)] + np.linalg.norm(
                        np.array(self.goal) - np.array(new_node)
                    )
                    break

        # Visualize the entire tree after planning
        self.visualize_tree_bullet()

        # Final visualization with the optimal path
        if tuple(self.goal) in self.parents:
            # Remove all tree-related debug items
            for debug_id in self.debug_items:
                p.removeUserDebugItem(debug_id)

            # Visualize only the optimal path
            path = self.construct_path(self.goal)
            self.visualize_path_bullet(path)
            return path
        else:
            raise RuntimeError("Failed to find a path to the goal.")

    def construct_path(self, node):
        """
        Construct the path from the goal node to the start node.

        Args:
            node (np.array): The goal node.

        Returns:
            list: A list of nodes forming the path from start to goal.

        Raises:
            KeyError: If a node does not have a valid parent.
        """
        path = [node]

        while tuple(path[-1]) in self.parents:
            parent = self.parents[tuple(path[-1])]
            if parent is None:
                break
            path.append(parent)

        # Reverse the path to start from the start node
        path.reverse()

        # Debugging output for path validation
        if self.debug:
            print(f"Constructed path: {path}")

        return path

    def create_ref_from_path(self, path, sample_distance=0.1):
        """
        Create a smooth reference path by interpolating along the given path.

        Args:
            path (list): A list of nodes forming the path (as tuples or arrays).
            sample_distance (float): The distance between interpolated points.

        Returns:
            list: A list of interpolated points forming a smooth path.
        """
        reference_path = []

        # Ensure path is a list of numpy arrays
        path = [np.array(p) for p in path]

        # Iterate through segments in the path
        for i in range(len(path) - 1):
            segment_direction = path[i + 1] - path[i]
            segment_length = np.linalg.norm(segment_direction)

            # Normalize the direction vector
            unit_direction = segment_direction / segment_length

            # Interpolate points along the segment
            num_samples = int(segment_length / sample_distance)
            for j in range(num_samples):
                reference_point = path[i] + j * sample_distance * unit_direction
                reference_path.append(reference_point)

        # Add the final point to ensure the path ends at the goal
        reference_path.append(path[-1])

        return reference_path

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
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Begin RRT(*) algorithm here ####

    obstacle_ids = []
    # Plan path using RRT
    rrt_star = RRT_STAR(start, goal, obstacles, obstacle_ids, bounds, step_size=0.4, max_iter=1000, debug=True)
    path = rrt_star.plan()
    reference_path = rrt_star.create_ref_from_path(path)
    p.removeAllUserDebugItems()
    if path is not None:
        for j in range(len(path) - 1):
                p.addUserDebugLine(path[j], path[j + 1], [0, 0.6, 0], 2)

    print("RRT path found:", path)
    
   
    #### Initialize the logger #################################
    # logger = Logger(logging_freq_hz=control_freq_hz,
    #                 num_drones=num_drones,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 )

    #### Initialize the controllers ############################
    # if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    k = 0
    time_const = 0.1
    # Follow the planned RRT path
    for target in reference_path:
        k = k + 1
        if debug:
            p.addUserDebugText("Target", target, textColorRGB=[0, 1, 0], textSize=1.2)

        # Move towards each target waypoint
        for step in range(0, int(time_const*env.CTRL_FREQ)):  # Adjust loop for smooth movement
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
            
            env.render()
            sync(step + (k-1)*int(time_const*env.CTRL_FREQ), START, env.CTRL_TIMESTEP)
        #### Log the simulation ####################################
        # for j in range(num_drones):
        #     logger.log( drone=j,
        #                 timestamp=i / env.CTRL_FREQ,
        #                 state=obs[j],
        #                 control=np.hstack([target, np.zeros(9)])  # Logging the target position with padding for consistency
        #                 )

        #### Printout ##############################################
            #

        #### Sync the simulation ###################################
        # if gui:
        #     sync(i, START, env.CTRL_TIMESTEP)

            

    #### Close the environment #################################
    env.close()

    # #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("rrt") # Optional CSV save

    # #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    # Define and parse (optional) arguments for the script
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