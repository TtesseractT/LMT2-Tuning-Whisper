import os
import subprocess

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
