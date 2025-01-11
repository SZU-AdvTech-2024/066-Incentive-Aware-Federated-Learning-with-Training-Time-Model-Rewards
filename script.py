import os
import sys
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_python_script(yaml_file):
    print(f"*****************************************************************************************")
    print(f"Running the Python script with the config file: {yaml_file}")
    print(f"*****************************************************************************************")
    result = subprocess.run(["python", "main_fl.py", "--config", yaml_file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_dir> [-g] [--max_workers N]")
        sys.exit(1)

    config_dir = sys.argv[1]
    generate_config = "-g" in sys.argv
    max_workers = None

    if "--max_workers" in sys.argv:
        max_workers_index = sys.argv.index("--max_workers") + 1
        if max_workers_index < len(sys.argv):
            max_workers = int(sys.argv[max_workers_index])

    print(config_dir)

    if generate_config:
        print(f"Generating the config file with the config directory: {config_dir}")
        result = subprocess.run(["python", "generate_config.py", "--path", config_dir], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("Config file generated successfully!")

    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    if not yaml_files:
        print(f"No .yaml files found in the directory {config_dir}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_python_script, yaml_files)

if __name__ == "__main__":
    main()
