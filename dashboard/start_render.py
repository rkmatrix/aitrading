#!/usr/bin/env python3
"""
Universal Render startup script - works regardless of Root Directory setting
Detects environment and imports correctly
"""
import sys
import os
from pathlib import Path

# Get current directory (where this script is located)
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Check if dashboard/ exists as subdirectory (Root Directory = empty)
# or if we're already in the copied location (Root Directory = dashboard/)
has_dashboard_subdir = (parent_dir / 'dashboard').exists() and (parent_dir / 'dashboard' / 'app.py').exists()

if has_dashboard_subdir:
    # Root Directory = empty - we're in /opt/render/project/src/dashboard/
    # Need to go to parent and import from dashboard
    sys.path.insert(0, str(parent_dir))
    os.chdir(parent_dir)
    from dashboard.app import app, socketio
else:
    # Root Directory = dashboard/ - files are copied directly to /opt/render/project/src/
    # Current dir is /opt/render/project/src/ (contains app.py, config.py, etc. directly)
    # But app.py has imports like "from dashboard.config" which won't work!
    # We need to create a dashboard module or fix imports
    sys.path.insert(0, str(current_dir))
    os.chdir(current_dir)
    
    # Try importing directly first
    try:
        from app import app, socketio
    except ImportError as e:
        # If that fails, app.py probably has "from dashboard.X" imports
        # We need to add current_dir to path as 'dashboard' module
        import importlib.util
        # Create a fake dashboard module by adding current_dir to path
        # and creating __init__.py if needed
        dashboard_init = current_dir / '__init__.py'
        if not dashboard_init.exists():
            dashboard_init.write_text('# Dashboard module\n')
        
        # Now try importing again - Python will treat current_dir as dashboard module
        # But wait, that won't work because imports are absolute
        # The real fix: app.py needs to use relative imports or we need Root Directory = empty
        # For now, let's try the run.py approach which changes directory
        print(f"Direct import failed: {e}")
        print("Trying alternative approach...")
        # Change to parent and use run.py logic
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        os.chdir(parent_dir)
        from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
