#!/usr/bin/env python3
"""
AI Shift Studio - Windows Launcher
Starts the Streamlit app in a native desktop window using webview
"""

import os
import sys
import threading
import time
import socket
from pathlib import Path

def find_free_port(start_port=8501, max_tries=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_tries}")

def start_streamlit_server(port, application_path):
    """Start Streamlit server in a background thread"""
    # Patch Streamlit to skip signal handler setup
    import streamlit.web.bootstrap as bootstrap
    original_set_up_signal_handler = bootstrap._set_up_signal_handler

    def patched_signal_handler(*args, **kwargs):
        """Skip signal handler setup when in background thread"""
        pass

    bootstrap._set_up_signal_handler = patched_signal_handler

    from streamlit.web import cli as stcli

    # Don't change directory - we want to stay in the writable data directory
    # The application_path is only used to find app.py

    # Set Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        str(application_path / "app.py"),
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--global.developmentMode", "false"
    ]

    # Run Streamlit (this blocks)
    stcli.main()

def wait_for_server(port, timeout=30):
    """Wait for the Streamlit server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False

def main():
    """Main launcher function - runs webview in main thread"""
    try:
        # Get the directory where the app files are located
        if getattr(sys, 'frozen', False):
            # When frozen, PyInstaller extracts files to sys._MEIPASS temp directory
            application_path = Path(sys._MEIPASS)

            # Set working directory to user's home directory (writable)
            # User can save files anywhere using file dialogs
            os.chdir(Path.home())
        else:
            application_path = Path(__file__).parent
            os.chdir(application_path)

        # Find a free port
        port = find_free_port()

        # Set environment variables for Streamlit
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

        print(f"Starting AI Shift Studio on port {port}...")

        # Start Streamlit in a background thread
        streamlit_thread = threading.Thread(
            target=start_streamlit_server,
            args=(port, application_path),
            daemon=True
        )
        streamlit_thread.start()

        # Wait for server to be ready
        print("Waiting for server to start...")
        if not wait_for_server(port, timeout=30):
            print("\nError: Server failed to start")
            input("\nPress Enter to exit...")
            sys.exit(1)

        print("Server is ready!")

        # Open webview in the MAIN thread (required for webview to work)
        import webview
        print("Opening native application window...")

        # Create and show the window in main thread
        webview.create_window(
            'AI Shift Studio',
            f'http://localhost:{port}',
            width=1400,
            height=900,
            resizable=True,
            fullscreen=False,
            min_size=(800, 600)
        )

        # Start webview - this MUST be in main thread and blocks until window closes
        webview.start()

        print("\nShutting down...")

    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
