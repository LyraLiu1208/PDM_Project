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
        #self._add_static_box(position, shelf_size (width, depth, height), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([0,0,0.3]), np.array([4, 0.2, 0.6]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([0,0,1.2]), np.array([4, 0.2, 0.6]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([0,0,0.75]), np.array([2, 0.2, 0.6]), color=[0.5, 0.5, 0.5])  # Gray color


        self._add_static_box(np.array([0,1.5,0.2]), np.array([4, 0.2, 0.4]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([0,1.5,1.15]), np.array([4, 0.2, 0.7]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([-1.5,1.5,0.75]), np.array([1, 0.2, 0.9]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([1.5,1.5,0.75]), np.array([1, 0.2, 0.9]), color=[0.5, 0.5, 0.5])  # Gray color

        self._add_static_box(np.array([0,3,0.75]), np.array([1, 0.2, 1.5]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([-1.5,3,0.75]), np.array([1, 0.2, 1.5]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([1.5,3,0.75]), np.array([1, 0.2, 1.5]), color=[0.5, 0.5, 0.5])  # Gray color

        self._add_static_box(np.array([-1, 3,1.3]), np.array([1, 0.2, 0.4]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([1,3,0.2]), np.array([1, 0.2, 0.4]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([1,3,1.2]), np.array([1, 0.2, 0.6]), color=[0.5, 0.5, 0.5])  # Gray color


        self._add_static_box(np.array([-2,1.5,0.75]), np.array([0.2, 3, 1.5]), color=[0.5, 0.5, 0.5])  # Gray color
        self._add_static_box(np.array([2,1.5,0.75]), np.array([0.2, 3, 1.5]), color=[0.5, 0.5, 0.5])  # Gray color
    
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