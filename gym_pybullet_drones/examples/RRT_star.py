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
            kd_tree (scipy.spatial.KDTree): KD-Tree for efficient collision detection.
            costs (dict): Dictionary storing the cost to reach each node.
            parents (dict): Dictionary storing parent-child relationships for tree nodes.
            children (defaultdict): Dictionary mapping nodes to their children.
            radius_const (float): Constant used to calculate neighbor search radius.
            edges (dict): Dictionary storing edges between nodes in the tree for connection tracking.
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles  # Obstacles are represented as (center, size) tuples
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        
        self.tree = [self.start]
        self.path = []
        self.debug = debug
        self.obstacle_ids = obstacle_ids
        
        # Initialize KDTree for obstacle centers for efficient nearest neighbor queries
        self.obstacle_centers = np.array([obs[0] for obs in obstacles])  # Extract obstacle centers
        self.kd_tree = KDTree(self.obstacle_centers)  # Build KDTree for fast spatial indexing

        # Initialize RRT* attributes
        self.costs = {tuple(self.start): 0}  # Start node cost is 0
        self.parents = {tuple(self.start): None}  # Start node has no parent
        self.children = defaultdict(list)  # Tracks children of each node
        self.radius_const = 3  # Used for neighbor search radius

        # Initialize edges attribute for tracking connections
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
        nearby_indices = self.kd_tree.query_ball_point(point, collision_threshold)

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

    def get_random_point(self):
        """Generate a random point within bounds."""
        while True:
            point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            if not self.is_in_collision(point):
                return point



    def nearest_neighbor(self, point):
        """Find the nearest neighbor in the tree to the given point."""
        distances = []
        for i in range(len(self.tree)):
            #distance = [np.linalg.norm(point - self.tree[i])]
            distances.append(np.linalg.norm(point - self.tree[i]))
        indices = np.argsort(distances)

        for index in indices:
            if not self.edge_in_collision(self.tree[int(index)], point):
                return self.tree[int(index)]
        return None
    


    def steer(self, from_node, to_node):
        """Steer from one node towards another by step size."""
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            if self.debug:
                print(f"Direct connection: {from_node} to {to_node}")
            return to_node
        return from_node + (direction / distance) * self.step_size
    


    def near(self, node):
        """Find all nodes within given radius from new node"""
        return [n for n in self.tree if np.linalg.norm(n - node) < self.radius] #For config space change distance metric



    def cost(self, node):
        """Return the cost of reaching a node from start"""
        return self.costs.get(tuple(node), float('inf'))
    


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
                

    def plan(self):
        """Plan a path using RRT."""
        best_path = None
        best_cost = float('inf')
        self.costs[tuple(self.goal)] = float('inf')
        i = 0
        b = 0
        N = 2
        goal_reached = False
        while i < self.max_iter or not best_path:
            i = i + 1
            #print(f"i = {i}, N = {N}")
            
            # if i%100 == 0:
            #     p.removeAllUserDebugItems()
            #     for debugnode in self.tree:
            #         if np.any(self.parents[tuple(debugnode)] != None):
            #             p.addUserDebugLine(self.start, self.start + [0, 0, 0.05], [1, 0, 1], 10)
            #             p.addUserDebugLine(self.goal, self.goal + [0, 0, 0.05], [0, 1, 0], 10)
            #             p.addUserDebugLine(debugnode, self.parents[tuple(debugnode)], [1, 0, 0], 3)
            #     while True:
            #         user_input = input("continue??? y")
            #         if user_input.lower() == "y":
            #             break


            self.radius = self.radius_const*(np.log(N)/N)**(1/3)
            rand_point = self.get_random_point()        #Returns randiom collision free point (collision check is done inside the function call)
            best_neigbor = self.lowest_cost_neighbor(rand_point) #Return closest neighbour point for rand_point
            #new_point = self.steer(nearest, rand_point)
            new_point = rand_point
            #p.addUserDebugText(".", new_point, textColorRGB=[0, 1, 0], textSize=2)
            
            if best_neigbor is not None:
                
                N = N + 1
                #print(f"\n\n radius is: {self.radius} m \n\n")
                self.tree.append(new_point)
                #self.parents[tuple(new_point)] = best_neigbor
                self.alter_parent(new_point, best_neigbor)
                self.costs[tuple(new_point)] = self.cost(best_neigbor) + np.linalg.norm(best_neigbor - new_point)
                self.rewire(new_point)

                #self.costs[tuple(self.goal)] = 0
                if goal_reached:
                    path = [self.goal]
                    while path[-1] is not None:
                        path.append(self.parents[tuple(path[-1])])
                    path = path[::-1][2:]
                
                    for node in path:
                        self.costs[tuple(node)] = self.costs[tuple(self.parents[tuple(node)])] + np.linalg.norm(node - self.parents[tuple(node)])
                
                
                if not self.edge_in_collision(new_point, self.goal): # and best_cost > 1000:
                    if self.cost(new_point) + np.linalg.norm(new_point - self.goal) < self.costs[tuple(self.goal)]:
                        goal_reached = True
                        #self.tree.append(self.goal)
                        #self.parents[tuple(self.goal)] = new_point
                        self.alter_parent(self.goal, new_point)
                        self.costs[tuple(self.goal)] = self.cost(new_point) + np.linalg.norm(new_point - self.goal)

                
                if self.cost(self.goal) < best_cost:
                    best_cost = self.cost(self.goal)
                    best_path = self.construct_path(b)
                    b = b+1
            #else:
                #print("No neighbour possible \n")
            # Print progress at every 100 iterations
            if i % 50 == 0:
                print(f"\n\n\n\n Iteration {i}: Tree size = {len(self.tree)} \n\n\n\n")

        if best_path:
            #time.sleep(10)
            return best_path
        else:
            print("Failed to find a path!")
            return []



    def construct_path(self, b):
        """Construct the path from goal to start."""
        path = [self.goal]
        if b > 6:
            b = 6
        while path[-1] is not None:
            path.append(self.parents[tuple(path[-1])])

        if path is not None:
            for j in range(len(path) - 2):
                if b > 3:
                    p.addUserDebugLine(path[j], path[j + 1], [1, 0.33*(b-3), 1], 5)
                else:
                    p.addUserDebugLine(path[j], path[j + 1], [0.33*b, 0, 1], 5)
        return path[::-1][1:]


    def create_ref_from_path(self, path):
            sample_distance = 0.05
            ref_path = []
            for i in range(len(path)-1):
                segment_direction = path[i+1] - path[i]
                segment_length = np.linalg.norm(segment_direction)

                steps = round(segment_length/sample_distance)
                ref_path.append(path[i])
                for step in range(steps-1):
                    ref_path.append(path[i]+step/steps*segment_direction)
            return ref_path

        # path = [self.goal]
        # while True:
        #     for i, item in enumerate(self.tree):
        #         if np.array_equal(item, path[-1]):
        #             del self.tree[i]
        #             break
        #     nearest = self.nearest_neighbor(path[-1])
        #     if np.linalg.norm(nearest - self.start) < 1e-2:
        #         path.append(self.start)
        #         break
        #     path.append(nearest)
        # if self.debug:
        #     print(f"Constructed path: {path}")
        # return path[::-1]


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