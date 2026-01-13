"""
Bot Controller - Manages bot lifecycle and state
"""
import subprocess
import psutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BotController:
    """Controls bot start/stop and monitors status."""
    
    def __init__(self):
        self.bot_process: Optional[subprocess.Popen] = None
        self.bot_script = Path("runner/phase26_realtime_live.py")
        self.kill_switch_path = Path("data/runtime/trading_disabled.flag")
        self.state_file = Path("data/runtime/bot_state.json")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        is_running = self._is_running()
        kill_switch_active = self.kill_switch_path.exists()
        
        status = {
            "status": "running" if is_running else "stopped",
            "kill_switch_active": kill_switch_active,
            "pid": self.bot_process.pid if self.bot_process else None,
            "uptime_seconds": self._get_uptime() if is_running else 0,
            "last_activity": self._get_last_activity(),
        }
        
        return status
    
    def start(self) -> Dict[str, Any]:
        """Start the bot."""
        if self._is_running():
            return {"success": False, "message": "Bot is already running"}
        
        try:
            self.bot_process = subprocess.Popen(
                ["python", str(self.bot_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            self._save_state({"status": "running", "started_at": datetime.now().isoformat()})
            
            return {"success": True, "message": "Bot started", "pid": self.bot_process.pid}
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            return {"success": False, "message": str(e)}
    
    def stop(self) -> Dict[str, Any]:
        """Stop the bot."""
        if not self._is_running():
            return {"success": False, "message": "Bot is not running"}
        
        try:
            if self.bot_process:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=10)
                self.bot_process = None
            
            self._save_state({"status": "stopped", "stopped_at": datetime.now().isoformat()})
            
            return {"success": True, "message": "Bot stopped"}
        except Exception as e:
            logger.error(f"Failed to stop bot: {e}")
            return {"success": False, "message": str(e)}
    
    def toggle_kill_switch(self, activate: bool) -> Dict[str, Any]:
        """Activate or deactivate kill switch."""
        try:
            if activate:
                data = {
                    "kill": True,
                    "enabled": True,
                    "status": "halted",
                    "activated_at": datetime.now().isoformat(),
                }
                self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
                self.kill_switch_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return {"success": True, "message": "Kill switch activated"}
            else:
                if self.kill_switch_path.exists():
                    self.kill_switch_path.unlink()
                return {"success": True, "message": "Kill switch deactivated"}
        except Exception as e:
            logger.error(f"Failed to toggle kill switch: {e}")
            return {"success": False, "message": str(e)}
    
    def _is_running(self) -> bool:
        """Check if bot process is running."""
        if self.bot_process:
            return self.bot_process.poll() is None
        
        # Check by script name
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'phase26_realtime_live.py' in ' '.join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return False
    
    def _get_uptime(self) -> float:
        """Get bot uptime in seconds."""
        if not self._is_running():
            return 0.0
        
        try:
            state = self._load_state()
            if state and "started_at" in state:
                started = datetime.fromisoformat(state["started_at"])
                return (datetime.now() - started).total_seconds()
        except Exception:
            pass
        
        return 0.0
    
    def _get_last_activity(self) -> Optional[str]:
        """Get last activity timestamp."""
        try:
            log_file = Path("data/logs/phase26_realtime_live.log")
            if log_file.exists():
                # Read last line
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        # Extract timestamp from last line
                        last_line = lines[-1]
                        # Parse timestamp (format: 2026-01-10 07:20:32,246)
                        parts = last_line.split('|')
                        if len(parts) > 0:
                            return parts[0].strip()
        except Exception:
            pass
        
        return None
    
    def _save_state(self, state: Dict[str, Any]) -> None:
        """Save bot state."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load bot state."""
        try:
            if self.state_file.exists():
                return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        
        return None
