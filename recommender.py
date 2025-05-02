import os
import sys
import subprocess
import time
import webbrowser
import requests

def check_python_version():
    """Check if Python version is at least 3.6"""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)

def create_directory_structure():
    """Create the necessary directories if they don't exist"""
    directories = ['models', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'flask', 'requests']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing missing packages: " + ", ".join(missing_packages))
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("All required packages installed successfully!")
    else:
        print("All required packages are already installed.")

def check_files():
    """Check if all required files exist"""
    required_files = [
        'data_preparation.py',
        'recommendation_models.py',
        'app.py',
    ]
    
    templates_files = [
        'templates/index.html',
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for file in templates_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    else:
        print("All required files exist.")

def wait_for_flask(port=5000, max_attempts=10):
    """Wait for the Flask app to start"""
    url = f"http://127.0.0.1:{port}/"
    
    print(f"Waiting for the Flask app to start at {url}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Flask app is running!")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt + 1}/{max_attempts}: Flask app not ready yet...")
            time.sleep(2)
    
    print("Error: Failed to connect to the Flask app after multiple attempts.")
    return False

def main():
    """Main function to run the movie recommender system"""
    print("=== Movie Recommender System Setup ===")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directory_structure()
    
    # Check required packages
    check_requirements()
    
    # Check required files
    check_files()
    
    # Run the Flask app
    print("\n=== Starting Movie Recommender System ===")
    print("This will download the MovieLens dataset (~1MB) and train the recommendation models.")
    print("Please wait while the system initializes...")
    
    # Start Flask app in a subprocess
    flask_process = subprocess.Popen([sys.executable, "app.py"])
    
    # Wait for the Flask app to start
    if wait_for_flask():
        # Open web browser
        webbrowser.open("http://127.0.0.1:5000/")
        
        print("\n=== System is running! ===")
        print("The web interface is now open in your browser.")
        print("Press Ctrl+C to stop the server when you're done.")
        
        try:
            # Keep the script running until Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            flask_process.terminate()
            print("System stopped.")
    else:
        # Terminate Flask process if it failed to start
        flask_process.terminate()
        print("Failed to start the system. Please check for errors.")

if __name__ == "__main__":
    main()