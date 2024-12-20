import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class CollapsedBuildingEnvironment(CtrlAviary):
    def __init__(self, **kwargs):
        """
        Initialize the warehouse environment with custom obstacles.
        """
        super().__init__(**kwargs)
        self.obstacles = []

        self._add_static_obstacles()


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


    def get_obstacles(self):
        return self.obstacles