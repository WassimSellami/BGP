import subprocess
import time
import sys
import signal
import os

def signal_handler(sig, frame):
    print("\nShutting down servers...")
    # The child processes should be terminated when the parent process exits
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("Starting BGP Real-time Prediction System...")
    
    # Start backend server
    print("Starting backend server...")
    backend = subprocess.Popen([sys.executable, 'g3_backend.py'])
    
    # Wait a bit to ensure backend is up
    time.sleep(2)
    
    # Start frontend server
    print("Starting frontend server...")
    frontend = subprocess.Popen([sys.executable, 'g3_frontend.py'])
    
    print("\nServers are running!")
    print("Access the visualization at: http://localhost:5000")
    print("Press Ctrl+C to stop all servers")
    
    try:
        # Keep the main process running
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        # Ensure child processes are terminated
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()

if __name__ == "__main__":
    main() 