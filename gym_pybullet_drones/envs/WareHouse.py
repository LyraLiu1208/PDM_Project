import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class WarehouseEnvironment(CtrlAviary):
    def __init__(self, **kwargs):
        """
        Initialize the warehouse environment with custom obstacles.
        """
        super().__init__(**kwargs)
        self.obstacles = []
        self.dynamic_obstacles = []
        self.worker_paths = []  # To store predefined paths for workers
        self._add_warehouse_obstacles()

    def _add_warehouse_obstacles(self):
        """
        Add static shelves and dynamic obstacles to represent a warehouse environment.
        """
        # Define warehouse layout
        num_shelves_per_row = 4
        num_rows = 3
        shelf_size = np.array([0.4, 2.0, 1.0])  # Width, depth, height
        aisle_width = 1.0

        # Add shelves in rows and color them gray
        for row in range(num_rows):
            row_y = row * (shelf_size[1] + aisle_width)
            for shelf in range(num_shelves_per_row):
                x_position = shelf * (shelf_size[0] + aisle_width)
                position = np.array([x_position, row_y, shelf_size[2] / 2])
                self._add_static_box(position, shelf_size, color=[0.5, 0.5, 0.5])  # Gray color

        # Add dynamic obstacles (workers)
        worker_size = np.array([0.2, 0.2, 0.6])  # Width, depth, height
        worker_positions = [
            np.array([0.5, 0.5, worker_size[2] / 2]),  # Initial position for Worker 1
            np.array([3.5, 1.5, worker_size[2] / 2])   # Initial position for Worker 2
        ]
        worker_colors = [[1, 0, 0], [1, 1, 0]]  # Red and Yellow workers

        # Define fixed paths for workers
        self.worker_paths = [
            [  # Worker 1 (Red): Path between first and second aisle
                np.array([0.5, -1.5, worker_size[2] / 2]),  # Bottom-left of first aisle
                np.array([0.5, 8, worker_size[2] / 2]),  # Top-left of first aisle
                np.array([2, 8, worker_size[2] / 2]),  # Top-right of first aisle
                np.array([2, -1.5, worker_size[2] / 2]),  # Bottom-right of first aisle
                np.array([0.5, -1.5, worker_size[2] / 2])   # Loop back to start
            ],
            [  # Worker 2 (Yellow): Path between second and third aisle
                np.array([3.5, 1.5, worker_size[2] / 2]),  # Bottom-left of second aisle
                np.array([3.5, 7.5, worker_size[2] / 2]),  # Top-left of second aisle
                np.array([2.3, 7.5, worker_size[2] / 2]),  # Top-right of second aisle
                np.array([2.3, 1.5, worker_size[2] / 2]),  # Bottom-right of second aisle
                np.array([3.5, 1.5, worker_size[2] / 2])   # Loop back to start
            ]
        ]

        # Add workers and assign them to paths
        for position, color, path in zip(worker_positions, worker_colors, self.worker_paths):
            dynamic_id = self._add_dynamic_worker(position, worker_size, color)
            self.dynamic_obstacles.append((dynamic_id, path, 0))  # Start at the first point in the path

    def _add_static_box(self, position, size, color):
        """
        Helper to add a static box-shaped obstacle with a specific color.
        """
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=color + [1])  # Add transparency
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        self.obstacles.append((position, size))
        return body_id

    def _add_dynamic_worker(self, position, size, color):
        """
        Helper to add a dynamic worker (box-shaped obstacle) with a specific color.
        """
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=color + [1])  # Add transparency
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        return body_id

    def update_dynamic_obstacles(self):
        """
        Update the position of dynamic obstacles (workers) along their fixed paths.
        """
        for i, (obstacle_id, path, current_index) in enumerate(self.dynamic_obstacles):
            # Get the current and next positions
            current_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            current_pos = np.array(current_pos)
            next_index = (current_index + 1) % len(path)  # Loop back to start
            next_pos = path[next_index]

            # Calculate direction and move towards the next position
            direction = next_pos - current_pos
            if np.linalg.norm(direction) > 0:  # Avoid division by zero
                step = 0.05 * direction / np.linalg.norm(direction)  # Normalize and scale
            else:
                step = np.zeros(3)

            new_pos = current_pos + step

            # Check if the worker has reached the next position
            if np.linalg.norm(new_pos - next_pos) < 0.1:  # Threshold to consider reaching the waypoint
                new_pos = next_pos
                current_index = next_index  # Update to the next waypoint

            # Update the worker's position in the simulation
            p.resetBasePositionAndOrientation(obstacle_id, new_pos, [0, 0, 0, 1])

            # Save the updated state
            self.dynamic_obstacles[i] = (obstacle_id, path, current_index)

    def get_obstacles(self):
        """
        Return the list of static and dynamic obstacles.
        """
        return self.obstacles, self.dynamic_obstacles