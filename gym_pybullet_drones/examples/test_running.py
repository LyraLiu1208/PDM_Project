import subprocess
import time

# List of files to execute
files = [
    "baseline.py",
    "baseline_building.py",
    "RRT_star.py",
    "RRT_star_building.py"
]

# Number of repetitions for each file
repetitions = 5

# Function to run a file multiple times
def run_file(file_name, repetitions):
    for i in range(repetitions):
        print(f"Running {file_name}, trial {i + 1}/{repetitions}")
        try:
            # Start timing the execution
            start_time = time.time()
            
            # Execute the file as a subprocess
            subprocess.run(["python", file_name], check=True)
            
            # End timing
            elapsed_time = time.time() - start_time
            print(f"{file_name} completed in {elapsed_time:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {file_name}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

# Execute each file 5 times
for file in files:
    run_file(file, repetitions)

print("All tests completed.")