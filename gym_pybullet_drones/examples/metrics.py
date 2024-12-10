import yaml
import numpy as np
import os
from datetime import datetime

class Metrics:
    def __init__(self):
        self.data = {}

    def compute_path_length(self, path):
        """
        Compute the total length of the path.
        Args:
            path (list): List of waypoints.
        Returns:
            float: Total path length.
        """
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
        self.data['path_length'] = length
        return length

    def compute_computation_time(self, start_time, end_time):
        """
        Compute the total computation time.
        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
        Returns:
            float: Total computation time.
        """
        time_taken = end_time - start_time
        self.data['computation_time'] = time_taken
        return time_taken

    def compute_success_rate(self, results):
        """
        Compute the success rate based on multiple trials.
        Args:
            results (list): List of trial results as dictionaries with 'path_found' keys.
        Returns:
            float: Success rate as a percentage.
        """
        successes = 0
        for result in results:
            if result['path_found']:
                successes += 1
        
        success_rate = (successes / len(results)) * 100
        self.data['success_rate'] = success_rate
        return success_rate

    def compute_collision_free_rate(self, results):
        """
        Compute the rate of collision-free runs.
        Args:
            results (list): List of trial results as dictionaries with 'collided' keys.
        Returns:
            float: Collision-free rate as a percentage.
        """
        no_collisions = 0
        for result in results:
            if not result['collided']:
                no_collisions += 1
        
        collision_free_rate = (no_collisions / len(results)) * 100
        self.data['collision_free_rate'] = collision_free_rate
        return collision_free_rate

    def compute_smoothness(self, path):
        """
        Compute the smoothness of the path by analyzing angular deviations.
        Args:
            path (list): List of waypoints.
        Returns:
            float: Total angular deviation of the path.
        """
        smoothness = 0.0
        for i in range(1, len(path) - 1):
            vec1 = np.array(path[i]) - np.array(path[i - 1])
            vec2 = np.array(path[i + 1]) - np.array(path[i])
            angle = np.arccos(
                np.clip(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                    -1.0, 1.0
                )
            )
            smoothness += angle
        self.data['smoothness'] = smoothness
        return smoothness

    def save_to_yaml(self):
        """
        Save the metrics data to a YAML file.
        Args:
            filename (str): The name of the YAML file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join("Metrices", f"metrics_{timestamp}.yaml")
        with open(filename, 'w') as file:
            yaml.dump(self.data, file, default_flow_style=False)
        print(f"Metrics saved to {filename}")