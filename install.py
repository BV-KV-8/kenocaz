import subprocess
import sys
import os

def install():
    print("--- Keno Vision Dependency Installer ---")

    # Define the requirements file
    req_file = "requirements.txt"

    # Check if requirements.txt exists, if not, create it dynamically
    if not os.path.exists(req_file):
        print(f"Creating {req_file}...")
        with open(req_file, "w") as f:
            f.write("numpy\n")
            f.write("requests\n")
            f.write("tensorflow\n")

    print("Installing packages... (This may take a few minutes for TensorFlow)")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("\n✅ Success! All dependencies installed.")
        print("You can now run: python start.py")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing packages: {e}")
        print("Try running manually: pip install -r requirements.txt")

if __name__ == "__main__":
    install()
