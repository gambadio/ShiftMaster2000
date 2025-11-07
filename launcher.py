#!/usr/bin/env python3
"""
AI Shift Studio - Windows Launcher
Starts the Streamlit app in a native desktop window using webview
"""

import os
import sys

# CRITICAL: Must be at the very top before any other imports
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Prevent re-execution when frozen
    if getattr(sys, 'frozen', False):
        if os.environ.get('AI_SHIFT_STUDIO_RUNNING') == '1':
            sys.exit(0)
        os.environ['AI_SHIFT_STUDIO_RUNNING'] = '1'

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

def start_streamlit_thread(port):
    """Start Streamlit server in a background thread (for frozen apps)"""
    from streamlit.web import cli as stcli
    import sys
    
    # Get the directory where the executable/script is located
    if getattr(sys, 'frozen', False):
        application_path = Path(sys.executable).parent
    else:
        application_path = Path(__file__).parent
    
    os.chdir(application_path)
    
    # Set Streamlit to use the specified port
    sys.argv = [
        "streamlit",
        "run",
        str(application_path / "app.py"),
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    # Run Streamlit in this thread
    sys.exit(stcli.main())

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
    """Main launcher function"""
    try:
        # Find a free port
        port = find_free_port()
        
        # Start Streamlit in a background thread
        print(f"Starting AI Shift Studio on port {port}...")
        streamlit_thread = threading.Thread(
            target=start_streamlit_thread,
            args=(port,),
            daemon=True
        )
        streamlit_thread.start()
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        if not wait_for_server(port, timeout=30):
            print("Error: Server failed to start")
            return
        
        print("Server is ready!")
        
        # Try to use webview for native window, fallback to browser if not available
        try:
            import webview
            print("Opening native application window...")
            
            # Create and show the window
            window = webview.create_window(
                'AI Shift Studio',
                f'http://localhost:{port}',
                width=1400,
                height=900,
                resizable=True,
                fullscreen=False,
                min_size=(800, 600)
            )
            
            # Start the webview (this blocks until window is closed)
            webview.start()
            
        except ImportError:
            print("webview not available, opening in browser...")
            import webbrowser
            webbrowser.open(f'http://localhost:{port}')
            
            print("\nAI Shift Studio is running!")
            print("Press Ctrl+C to stop the application.\n")
            
            # Keep the main thread running
            try:
                while streamlit_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        print("\nShutting down...")
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
