#!/usr/bin/env python3
"""
Render-compatible app wrapper
Handles imports correctly when Root Directory is dashboard/
"""
import sys
import os
from pathlib import Path

# Detect if we're running with Root Directory = dashboard/
# (files copied directly) or Root Directory = empty (dashboard/ subdirectory exists)
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Check if dashboard/ exists as subdirectory
has_dashboard_subdir = (parent_dir / 'dashboard').exists() and (parent_dir / 'dashboard' / 'app.py').exists()

if not has_dashboard_subdir:
    # Root Directory = dashboard/ - files are copied directly
    # We're in /opt/render/project/src/ with app.py, config.py, etc. directly here
    # But app.py has "from dashboard.config" imports which won't work
    # Solution: Create a dashboard module by adding current_dir to path
    # and creating a symlink or modifying sys.modules
    
    # Add current directory to path
    sys.path.insert(0, str(current_dir))
    
    # Create a 'dashboard' module that points to current directory
    # This allows "from dashboard.config" to work
    import importlib.util
    import types
    
    # Create a module object for 'dashboard'
    dashboard_module = types.ModuleType('dashboard')
    dashboard_module.__path__ = [str(current_dir)]
    dashboard_module.__file__ = str(current_dir / '__init__.py')
    
    # Add it to sys.modules BEFORE importing app
    sys.modules['dashboard'] = dashboard_module
    
    # Now import app - it will use the dashboard module we just created
    os.chdir(current_dir)
    from app import app, socketio
else:
    # Root Directory = empty - normal structure
    sys.path.insert(0, str(parent_dir))
    os.chdir(parent_dir)
    from dashboard.app import app, socketio

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    print(f"Starting dashboard on port {port} (debug={debug_mode})")
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
