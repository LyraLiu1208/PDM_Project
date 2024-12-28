import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from collections import defaultdict
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
        self.debug = True
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

        self.metrics = Metrics()  # Initialize Metrics class for performance evaluation

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
        """
        Rewire the tree by updating parent relationships for nodes near the new_node.

        Args:
            new_node (np.array): The recently added node to the tree.

        """
        # Calculate the rewiring radius dynamically based on tree size
        radius = self.radius_const * (np.log(len(self.tree)) / len(self.tree)) ** (1 / 3)

        # Find neighbors using KDTree
        if not self.node_kd_tree or len(self.tree) != self.node_kd_tree.n:
            self.node_kd_tree = KDTree(self.tree)
        neighbor_indices = self.node_kd_tree.query_ball_point(new_node, radius)
        neighbors = [self.tree[idx] for idx in neighbor_indices]

        # Attempt rewiring for each neighbor
        for neighbor in neighbors:
            if np.array_equal(neighbor, new_node):  # Skip self-rewiring
                continue
            if self.edge_in_collision(new_node, neighbor):  # Skip if edge is in collision
                if self.debug:
                    print(f"Skipping rewiring for neighbor {neighbor}: edge collision detected.")
                continue

            # Calculate cost through the new_node
            cost_via_new_node = (
                self.costs[tuple(new_node)] + np.linalg.norm(np.array(neighbor) - np.array(new_node))
            )

            # Update parent if rewiring reduces cost
            if cost_via_new_node < self.costs[tuple(neighbor)]:
                old_parent = self.parents.get(tuple(neighbor))
                if old_parent:
                    # Ensure synchronization of `self.children`
                    old_children = self.children.get(tuple(old_parent), [])
                    for i, child in enumerate(old_children):
                        if np.array_equal(child, neighbor):
                            self.children[tuple(old_parent)].pop(i)  # Remove old parent relationship
                            break
                    else:
                        if self.debug:
                            print(
                                f"Neighbor {neighbor} not found in children of {old_parent}. "
                                f"Current children: {self.children.get(tuple(old_parent), [])}"
                            )
                # Update parent and child relationships
                self.parents[tuple(neighbor)] = tuple(new_node)
                self.children[tuple(new_node)].append(neighbor)  # Add new parent relationship

                if self.debug:
                    print(
                        f"Rewired node {neighbor} from parent {old_parent} to {new_node} "
                        f"with new cost {cost_via_new_node}"
                    )
                
    def propagate_cost_updates(self, node):
        """
        Propagate cost updates to all descendants of the given node.

        Why it's needed:
            - Ensures the tree maintains valid cost values after rewiring or other updates.

        Args:
            node (np.array): The node whose descendants' costs need to be updated.
        """
        # Validate that the node exists in the tree
        if tuple(node) not in self.costs:
            raise ValueError(f"Node {node} is not in the tree or does not have a valid cost entry.")

        # Use a queue for breadth-first traversal
        queue = [node]

        while queue:
            current_node = queue.pop(0)
            children = self.children.get(tuple(current_node), [])

            for child in children:
                new_cost = self.costs[tuple(current_node)] + np.linalg.norm(
                    np.array(child) - np.array(current_node)
                )
                if abs(self.costs[tuple(child)] - new_cost) > 1e-6:
                    self.costs[tuple(child)] = new_cost
                    if self.debug:
                        print(f"Updated cost for node {child}: {new_cost}")
                    queue.append(child)
    
    def add_node(self, new_node, parent_node):
        """
        Add a new node to the tree and update parent-child relationships.

        Args:
            new_node (np.array): The new node to add.
            parent_node (np.array): The parent node of the new node.

        Updates:
            - Adds the new node to the tree.
            - Updates the parent in `self.parents`.
            - Updates the child relationships in `self.children`.
            - Calculates and stores the cost of the new node in `self.costs`.
        """
        # Add node to the tree
        self.tree.append(new_node)

        # Update parent-child relationships
        self.parents[tuple(new_node)] = tuple(parent_node)
        self.children[tuple(parent_node)].append(new_node)

        # Calculate and store the cost
        self.costs[tuple(new_node)] = self.costs[tuple(parent_node)] + np.linalg.norm(
            np.array(new_node) - np.array(parent_node)
        )

        # Debugging: Log node addition
        if self.debug:
            print(f"Added node {new_node} with cost {self.costs[tuple(new_node)]}, parent {parent_node}.")

    def children_of(self, node):
        """
        Find all children of the given node in the RRT* tree.

        Why it's needed:
            - Used in operations like cost propagation and rewiring.

        Args:
            node (np.array): The node whose children are to be found.

        Returns:
            list: A list of child nodes.
        """
        # Retrieve children from precomputed relationships
        children = self.children.get(tuple(node), [])

        if self.debug:
            if children:
                print(f"Children of node {node}: {children}")
            else:
                print(f"Node {node} has no children.")

            # Visualize children relationships in PyBullet
            for child in children:
                p.addUserDebugLine(
                    lineFromXYZ=node,
                    lineToXYZ=child,
                    lineColorRGB=[0, 1, 0],  # Green for child edges
                    lineWidth=2.0,
                    lifeTime=1.0  # Temporary visualization for debugging
                )

        return children

    def lowest_cost_neighbor(self, node, radius=None):
        """
        Find the neighbor that provides the lowest-cost path to the given node.

        Args:
            node (np.array): The node for which the lowest-cost neighbor is sought.
            radius (float, optional): The search radius. Defaults to a dynamic radius based on tree size.

        Returns:
            np.array: The neighbor providing the lowest-cost path, or None if no valid neighbor exists.
        """
        # Calculate a default radius if not provided
        if radius is None:
            radius = self.radius_const * (np.log(len(self.tree)) / len(self.tree)) ** (1 / 3)

        # Find neighbors using KDTree
        if not self.node_kd_tree or len(self.tree) != self.node_kd_tree.n:
            self.node_kd_tree = KDTree(self.tree)
        neighbor_indices = self.node_kd_tree.query_ball_point(node, radius)
        neighbors = [self.tree[idx] for idx in neighbor_indices]

        # Initialize variables for the lowest cost neighbor
        lowest_cost = float('inf')
        best_neighbor = None

        # Iterate through neighbors to find the one with the lowest cost
        for neighbor in neighbors:
            if np.array_equal(neighbor, node):  # Skip the node itself
                continue

            # Check if the edge is collision-free
            if self.edge_in_collision(node, neighbor):
                if self.debug:
                    print(f"Skipping neighbor {neighbor}: edge collision detected.")
                continue

            # Calculate the cost of reaching the node through this neighbor
            cost = self.costs[tuple(neighbor)] + np.linalg.norm(np.array(neighbor) - np.array(node))

            # Update the lowest-cost neighbor
            if cost < lowest_cost:
                lowest_cost = cost
                best_neighbor = neighbor

        # Debugging information
        if self.debug:
            if best_neighbor is not None:
                print(f"Lowest-cost neighbor for node {node}: {best_neighbor} with cost {lowest_cost}")
            else:
                print(f"No valid neighbors found for node {node} within radius {radius}")

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
        """
        Alter the parent of a node and update the tree structure.

        Args:
            node (np.array): The node whose parent is being altered.
            new_parent (np.array): The new parent node.
        """
        # Validate that the node exists in the tree
        if tuple(node) not in self.parents:
            raise ValueError(f"Node {node} is not in the tree or has no parent.")

        # Prevent redundant updates
        current_parent = self.parents[tuple(node)]
        if np.array_equal(current_parent, new_parent):
            if self.debug:
                print(f"No change: {node} already has {new_parent} as its parent.")
            return

        # Remove the node from its current parent's children
        if current_parent:
            old_children = self.children.get(tuple(current_parent), [])
            for i, child in enumerate(old_children):
                if np.array_equal(child, node):
                    self.children[tuple(current_parent)].pop(i)
                    break

        # Update the parent and child relationships
        self.parents[tuple(node)] = tuple(new_parent)
        self.children[tuple(new_parent)].append(node)

        # Recalculate the cost for the node
        self.costs[tuple(node)] = self.costs[tuple(new_parent)] + np.linalg.norm(
            np.array(node) - np.array(new_parent)
        )

        # Debugging: Log the parent alteration
        if self.debug:
            print(f"Altered parent of node {node} from {current_parent} to {new_parent}.")
            print(f"New cost for node {node}: {self.costs[tuple(node)]}")

        # Propagate cost updates to descendants
        self.propagate_cost_updates(node)
    
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

        Returns:
            list: A list of nodes forming the optimal path from start to goal.
        """
        self.debug_items = []  # Initialize debug visualization items for PyBullet
        start_time = time.time()  # Start timing

        for iteration in range(self.max_iter):
            # Sample a random point in the configuration space
            random_point = self.get_random_point()

            # Find the nearest node to the sampled point
            nearest_node = self.nearest_neighbor(random_point)

            # Steer towards the sampled point
            new_node = self.steer(nearest_node, random_point)

            # Check if the new node is collision-free and valid
            if not self.is_in_collision(new_node):
                # Add the new node to the tree and update relationships
                self.add_node(new_node, nearest_node)

                # Find the lowest-cost neighbor and alter the parent if it reduces the cost
                best_neighbor = self.lowest_cost_neighbor(new_node)
                if best_neighbor is not None and not np.array_equal(best_neighbor, nearest_node):
                    self.alter_parent(new_node, best_neighbor)

                # Rewire the tree to optimize paths
                self.rewire(new_node)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < self.step_size:
                    self.add_node(self.goal, new_node)
                    break

        end_time = time.time()  # End timing

        # Final visualization of the tree and optimal path
        if tuple(self.goal) in self.parents:
            # Visualize the entire tree only once the optimal path is found
            self.visualize_tree_bullet()

            # Construct and visualize the optimal path
            path = self.construct_path(self.goal)
            self.visualize_path_bullet(path)

            # Compute and log metrics
            self.metrics.compute_computation_time(start_time, end_time)
            self.metrics.compute_path_length(path)
            self.metrics.compute_smoothness(path)

            print(f"Path length: {self.metrics.data['path_length']}")
            print(f"Computation time: {self.metrics.data['computation_time']} seconds")
            print(f"Path smoothness: {self.metrics.data['smoothness']}")

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
        """
        path = [node]

        while tuple(path[-1]) in self.parents:
            parent = self.parents[tuple(path[-1])]
            if parent is None:  # Stop at the start node
                break
            path.append(parent)

        path.reverse()

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
        path = [np.array(p) for p in path]

        for i in range(len(path) - 1):
            segment_direction = path[i + 1] - path[i]
            segment_length = np.linalg.norm(segment_direction)

            unit_direction = segment_direction / segment_length
            num_samples = int(segment_length / sample_distance)

            for j in range(num_samples):
                reference_point = path[i] + j * sample_distance * unit_direction
                reference_path.append(reference_point)

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

    # Collect metrics for this trial
    trial_results = [{
        "trial": 1,
        "path_length": rrt_star.metrics.data.get("path_length", 0),
        "smoothness": rrt_star.metrics.data.get("smoothness", 0),
        "computation_time": rrt_star.metrics.data.get("computation_time", 0)
    }]

    # Save metrics to a YAML file
    rrt_star.metrics.save_to_yaml(folder="metrics/WareHouse/RRT_star", trial_results=trial_results)
    print("Metrics saved successfully.")

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