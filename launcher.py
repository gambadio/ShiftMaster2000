#!/usr/bin/env python3
"""
AI Shift Studio - Windows Launcher
Starts the Streamlit app and opens it in the default browser
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path

def main():
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        application_path = Path(sys.executable).parent
    else:
        # Running as script
        application_path = Path(__file__).parent

    # Change to application directory
    os.chdir(application_path)

    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    # Start Streamlit in a subprocess
    print("Starting AI Shift Studio...")
    streamlit_cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        "app.py",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    # Start Streamlit
    process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for Streamlit to start
    print("Waiting for server to start...")
    time.sleep(3)

    # Open browser
    url = "http://localhost:8501"
    print(f"Opening {url} in your browser...")
    webbrowser.open(url)

    print("\nAI Shift Studio is running!")
    print("Close this window to stop the application.\n")

    # Keep the process running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
