import yaml
import numpy as np
import os
from datetime import datetime

class Metrics:
    def __init__(self):
        self.data = {}

    def compute_path_length(self, path):
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
        self.data['path_length'] = float(length)
        return length

    def compute_computation_time(self, start_time, end_time):
        time_taken = end_time - start_time
        self.data['computation_time'] = time_taken
        return time_taken

    def compute_smoothness(self, path):
        """
        Compute the smoothness of the path by analyzing angular deviations.
        Args:
            path (list): List of waypoints as [x, y, z].
        Returns:
            float: Total angular deviation of the path.
        """
        if len(path) < 3:
            # A path with fewer than 3 points cannot have angular deviations
            return 0.0

        smoothness = 0.0
        for i in range(1, len(path) - 1):
            vec1 = np.array(path[i]) - np.array(path[i - 1])  # Vector from previous to current point
            vec2 = np.array(path[i + 1]) - np.array(path[i])  # Vector from current to next point

            # Calculate the angle between vec1 and vec2
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                angle = np.arccos(
                    np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                )
                smoothness += angle

        return smoothness

    def compute_avg_iteration_time(self, total_time, iterations):
        """
        Compute the average time per iteration.
        Args:
            total_time (float): Total computation time in seconds.
            iterations (int): Total number of iterations.
        Returns:
            float: Average iteration time in seconds.
        """
        avg_iteration_time = total_time / iterations if iterations > 0 else 0.0
        self.data['avg_iteration_time'] = avg_iteration_time
        return avg_iteration_time

    def compute_number_of_iterations(self, iterations):
        """
        Store the number of iterations.
        Args:
            iterations (int): Total number of iterations.
        """
        self.data['number_of_iterations'] = iterations
        return iterations

    def save_to_yaml(self, folder="metrics", trial_results=None):
        """
        Save per-trial metrics and overall statistics to a YAML file.
        Args:
            folder (str): The folder where the YAML file will be saved.
            trial_results (list): A list of trial-specific metrics.
        """
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Calculate averages
        total_path_length = sum(trial.get("path_length", 0) for trial in trial_results if trial.get("path_length") is not None)
        total_smoothness = sum(trial.get("path_smoothness", 0) for trial in trial_results if trial.get("smoothness") is not None)
        total_computation_time = sum(trial.get("computation_time", 0) for trial in trial_results)
        total_iterations = sum(trial.get("num_iterations", 0) for trial in trial_results)

        num_trials_with_path = len([trial for trial in trial_results if trial.get("path_length") is not None])

        avg_path_length = total_path_length / num_trials_with_path 
        avg_smoothness = total_smoothness / num_trials_with_path 
        avg_computation_time = total_computation_time / len(trial_results)
        avg_iteration_time = total_computation_time / total_iterations 
        avg_num_iterations = total_iterations / len(trial_results)

        # Prepare averages section
        self.data["averages"] = {
            "avg_path_length": round(float(avg_path_length), 2),
            "avg_smoothness": round(float(avg_smoothness), 2),
            "avg_computation_time": round(float(avg_computation_time), 2),
            "avg_iteration_time": round(float(avg_iteration_time), 2),
            "avg_num_iterations": round(float(avg_num_iterations), 2),
            "num_trials": len(trial_results)
        }

        # Prepare data for each trial
        self.data["trials"] = {}
        for trial in trial_results:
            trial_number = trial["trial"]
            self.data["trials"][f"Trial {trial_number}"] = {
                "path_length": round(float(trial["path_length"]), 2),
                "path_smoothness": round(float(trial["path_smoothness"]), 2),
                "computation_time": round(float(trial["computation_time"]), 2),
                "num_iterations": trial["num_iterations"],
                "avg_iteration_time": round(float(trial["avg_iteration_time"]), 2),
            }

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(folder, f"metrics_{timestamp}.yaml")

        # Save metrics to YAML file
        with open(filename, "w") as file:
            yaml.dump(self.data, file, default_flow_style=False)

        print(f"Metrics saved to {filename}")