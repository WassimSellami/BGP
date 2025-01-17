import subprocess
import time
import sys
import signal

def signal_handler(sig, frame):
    print("\nShutting down servers...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("Starting BGP Real-time Prediction System...")
    
    print("Starting backend server...")
    backend = subprocess.Popen([sys.executable, 'backend.py'])
    
    time.sleep(2)
    
    print("Starting frontend server...")
    frontend = subprocess.Popen([sys.executable, 'frontend.py'])
    
    print("\nServers are running!")
    print("Access the visualization at: http://localhost:5000")
    print("Press Ctrl+C to stop all servers")
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()

if __name__ == "__main__":
    main() 