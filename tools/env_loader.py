# tools/env_loader.py
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def ensure_env_loaded(path: str = ".env") -> None:
    """Force-load environment variables from .env file."""
    if not os.getenv("APCA_API_KEY_ID"):
        env_path = Path(path).resolve()
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info("Loaded environment variables from %s", env_path)
        else:
            logger.warning("Environment file not found at %s", env_path)


def validate_api_keys(mode: str = "PAPER", fail_fast: bool = True) -> Dict[str, Any]:
    """
    Validate that required API keys are present.
    
    Args:
        mode: Trading mode (PAPER or LIVE)
        fail_fast: If True, raise RuntimeError on missing keys. If False, return validation result.
    
    Returns:
        Dict with validation results:
        {
            "valid": bool,
            "missing_keys": list[str],
            "mode": str,
            "base_url": str
        }
    
    Raises:
        RuntimeError: If fail_fast=True and keys are missing
    """
    ensure_env_loaded()
    
    key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
    base_url = os.getenv("APCA_API_BASE_URL", "").strip()
    
    missing_keys = []
    if not key_id:
        missing_keys.append("APCA_API_KEY_ID")
    if not secret_key:
        missing_keys.append("APCA_API_SECRET_KEY")
    
    # Validate base_url or set default based on mode
    if not base_url:
        if mode.upper() == "LIVE":
            base_url = "https://api.alpaca.markets"
        else:
            base_url = "https://paper-api.alpaca.markets"
        logger.info("Using default base_url for mode %s: %s", mode, base_url)
    
    result = {
        "valid": len(missing_keys) == 0,
        "missing_keys": missing_keys,
        "mode": mode.upper(),
        "base_url": base_url,
        "key_id_present": bool(key_id),
        "secret_key_present": bool(secret_key),
    }
    
    if not result["valid"]:
        error_msg = (
            f"❌ Missing required API keys for {mode} mode: {', '.join(missing_keys)}. "
            f"Please set these in your .env file or environment variables."
        )
        logger.error(error_msg)
        
        if fail_fast:
            raise RuntimeError(error_msg)
    
    if result["valid"]:
        logger.info("✅ API keys validated successfully for %s mode", mode)
    
    return result