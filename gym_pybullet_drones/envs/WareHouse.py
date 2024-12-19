import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class WarehouseEnvironment(CtrlAviary):
    def __init__(self, include_static=True, include_dynamic=True, **kwargs):
        """
        Initialize the warehouse environment with custom obstacles.
        """
        super().__init__(**kwargs)
        self.obstacles = []
        self.dynamic_obstacles = []
        self.worker_paths = []  # To store predefined paths for workers

        if include_static:
            self._add_static_obstacles()
        if include_dynamic:
            self._add_dynamic_obstacles()

    def _add_static_obstacles(self):
        """
        Add static shelves to the warehouse.
        """
        num_shelves_per_row = 4
        num_rows = 3
        shelf_size = np.array([0.4, 2.0, 1.0])  # Width, depth, height
        aisle_width = 1.0

        for row in range(num_rows):
            row_y = row * (shelf_size[1] + aisle_width)
            for shelf in range(num_shelves_per_row):
                x_position = shelf * (shelf_size[0] + aisle_width)
                position = np.array([x_position, row_y, shelf_size[2] / 2])
                self._add_static_box(position, shelf_size, color=[0.5, 0.5, 0.5])  # Gray color

    def _add_dynamic_obstacles(self):
        """
        Add dynamic obstacles (workers) with predefined paths.
        """
        worker_size = np.array([0.2, 0.2, 0.6])  # Width, depth, height
        worker_positions = [
            np.array([0.5, 0.5, worker_size[2] / 2]),  # Worker 1
            np.array([3.5, 1.5, worker_size[2] / 2])   # Worker 2
        ]
        worker_colors = [[1, 0, 0], [1, 1, 0]]  # Red and Yellow workers

        self.worker_paths = [
            [np.array([0.5, -1.5, worker_size[2] / 2]),
             np.array([0.5, 8, worker_size[2] / 2]),
             np.array([2, 8, worker_size[2] / 2]),
             np.array([2, -1.5, worker_size[2] / 2]),
             np.array([0.5, -1.5, worker_size[2] / 2])],  # Loop for Worker 1
            [np.array([3.5, 1.5, worker_size[2] / 2]),
             np.array([3.5, 7.5, worker_size[2] / 2]),
             np.array([2.3, 7.5, worker_size[2] / 2]),
             np.array([2.3, 1.5, worker_size[2] / 2]),
             np.array([3.5, 1.5, worker_size[2] / 2])]   # Loop for Worker 2
        ]

        for position, color, path in zip(worker_positions, worker_colors, self.worker_paths):
            dynamic_id = self._add_dynamic_worker(position, worker_size, color)
            self.dynamic_obstacles.append((dynamic_id, path, 0))  # Initialize with path index 0

    def _add_static_box(self, position, size, color):
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=color + [1])
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        self.obstacles.append((position, size))
        return body_id

    def _add_dynamic_worker(self, position, size, color):
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=color + [1])
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        return body_id

    def update_dynamic_obstacles(self):
        for i, (obstacle_id, path, current_index) in enumerate(self.dynamic_obstacles):
            current_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            next_index = (current_index + 1) % len(path)
            next_pos = path[next_index]

            direction = next_pos - np.array(current_pos)
            if np.linalg.norm(direction) > 0:
                step = 0.01 * direction / np.linalg.norm(direction)
            else:
                step = np.zeros(3)

            new_pos = np.array(current_pos) + step
            if np.linalg.norm(new_pos - next_pos) < 0.1:
                new_pos = next_pos
                current_index = next_index

            p.resetBasePositionAndOrientation(obstacle_id, new_pos, [0, 0, 0, 1])
            self.dynamic_obstacles[i] = (obstacle_id, path, current_index)

    def get_obstacles(self):
        return self.obstacles, self.dynamic_obstacles