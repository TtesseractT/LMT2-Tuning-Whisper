import os
import torch
import subprocess

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device name:", torch.cuda.get_device_name(0))  # Prints the name of the first GPU
    print("CUDA device count:", torch.cuda.device_count())  # Prints the number of available GPUs
else:
    print("CUDA is not available.")

models_directory = "./Models"
initial_directory = os.getcwd()  # Store the initial working directory

# Get a list of subdirectories inside the Models directory
subdirectories = next(os.walk(models_directory))[1]

# Iterate through each subdirectory
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(models_directory, subdirectory)
    
    print(f"Entering subdirectory: {subdirectory}")
    
    # Change to the subdirectory
    os.chdir(subdirectory_path)
    
    # Run the TEST.py script
    os.system("python TRSC.py")
    
    print(f"Finished running TRSC.py in subdirectory: {subdirectory}")
    
    # Return to the initial working directory (Models)
    os.chdir(initial_directory)

print("All subdirectories processed.")
print("Processing Metrics")

models_folder = "./Models"
os.chdir(models_folder)

for root, dirs, files in os.walk(".", topdown=True):
    for name in dirs:
        subfolder_path = os.path.join(root, name)

        logs_folder = os.path.join(subfolder_path, "Logs")
        if not os.path.isdir(logs_folder):
            continue

        script_path = os.path.join(logs_folder, "Metric-calc.py")
        subprocess.run(["python", script_path], shell=True)