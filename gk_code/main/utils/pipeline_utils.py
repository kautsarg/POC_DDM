import os
import sys
from pathlib import Path
import config


def get_exp_paths(exp_folder):
    return sorted([
        Path(exp_folder, name)
        for name in os.listdir(exp_folder)
        if os.path.isdir(os.path.join(exp_folder, name)) and name not in config.EXCLUDED_FOLDERS
    ])


def check_task_id(task_id, paths):
    if task_id >= len(paths):
        print(f"Task ID {task_id} is out of bounds for {len(paths)} folders. Exiting.")
        sys.exit(0)
