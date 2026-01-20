import platform
import os
import sys
import psutil
import subprocess
import shutil

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def check_specs():
    print_section("System Information")
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")

    print_section("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    print_section("CPU Info")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    if cpufreq:
        print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
        print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
        print(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    print_section("Memory Information")
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

    print_section("Disk Information")
    # Get disk usage for current working directory drive
    cwd = os.getcwd()
    total, used, free = shutil.disk_usage(cwd)
    print(f"Drive ({os.path.splitdrive(cwd)[0]}):")
    print(f"  Total: {get_size(total)}")
    print(f"  Used: {get_size(used)}")
    print(f"  Free: {get_size(free)}")

    print_section("GPU Information")
    try:
        # Try nvidia-smi
        nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if nvidia_smi.returncode == 0:
            print("NVIDIA GPU detected:")
            print(nvidia_smi.stdout.strip())
        else:
            print("nvidia-smi check failed or no NVIDIA GPU found.")
    except FileNotFoundError:
        print("nvidia-smi not found.")

    print_section("Key Dependencies")
    deps = ['streamlit', 'google-cloud-aiplatform', 'langchain', 'fpdf', 'matplotlib', 'numpy', 'pandas']
    for dep in deps:
        try:
            dist = __import__('pkg_resources').get_distribution(dep)
            print(f"{dep}: {dist.version}")
        except Exception:
            print(f"{dep}: Not Found")
    
    print_section("Environment Variables (Existence Check)")
    env_vars = ['GOOGLE_APPLICATION_CREDENTIALS', 'PROJECT_ID', 'LOCATION']
    for var in env_vars:
        exists = "SET" if os.environ.get(var) else "NOT SET"
        print(f"{var}: {exists}")

if __name__ == "__main__":
    check_specs()
