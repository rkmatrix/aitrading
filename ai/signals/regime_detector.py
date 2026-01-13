# ai/signals/regime_detector.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RegimeThresholds:
    """
    Simple heuristic thresholds for macro regime detection.
    All values are configurable via YAML.
    """
    vix_risk_off: float = 25.0      # VIX >= this → risk-off
    vix_risk_on: float = 18.0       # VIX <= this → risk-on
    spx_ret_risk_off: float = -0.05 # 20d SPX return <= this → risk-off
    spx_ret_risk_on: float = 0.03   # 20d SPX return >= this → risk-on
    curve_inverted_bps: float = -10.0  # 10y-2y <= this (bps) → inversion / stress
    """
    Universal threshold structure compatible with:
    - Phase 26 runner
    - Phase 69C+ Regime Engine
    - Phase 77/84 Multi-Agent Fusion
    """

    # These are the ones Phase 26 passes:
    bull: float = 0.0
    bear: float = 0.0
    volatile: float = 0.0
    chop: float = 0.0

    # Legacy / older versions may use these:
    trend_up: float = 0.0
    trend_down: float = 0.0
    vol_high: float = 0.0
    vol_low: float = 0.0
    
    # NEW universal initializer
    def __init__(self, **kwargs):

        # Set ALL provided keys dynamically → handles high_vol, low_vol, chaos, extreme, etc.
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Fill missing known fields with defaults
        for f in [
            "bull", "bear", "volatile", "chop",
            "trend_up", "trend_down", "vol_high", "vol_low"
        ]:
            if not hasattr(self, f):
                setattr(self, f, 0.0)

        # FUTURE PROOFING:
        # If runner/code uses something like threshold.high_vol,
        # we auto-create it above from kwargs.

@dataclass
class RegimeDetectorConfig:
    enabled: bool = True
    # COMMON / KNOWN FIELDS
    window: int = 50                              # Phase 26 uses this
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Paths used in your existing RegimeDetector
    state_path: str = "data/regime/state.json"
    macro_features_path: str = "data/regime/macro.json"

    # Optional future fields
    volatility_lookback: int = 20
    trend_lookback: int = 20

    def __init__(self, **kwargs):
        # Apply all provided keys first (new → allowed)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Ensure required keys exist (fallback defaults)
        if not hasattr(self, "window"):
            setattr(self, "window", 50)

        if not hasattr(self, "thresholds"):
            setattr(self, "thresholds", {})

        if not hasattr(self, "state_path"):
            setattr(self, "state_path", "data/regime/state.json")

        if not hasattr(self, "macro_features_path"):
            setattr(self, "macro_features_path", "data/regime/macro.json")

        # Optional defaults
        for field_name, default_val in {
            "volatility_lookback": 20,
            "trend_lookback": 20,
        }.items():
            if not hasattr(self, field_name):
                setattr(self, field_name, default_val)



@dataclass
class RegimeState:
    name: str              # "RISK_ON" | "RISK_OFF" | "NEUTRAL"
    risk_level: str        # "HIGH" | "LOW" | "MEDIUM"
    score: float           # composite risk score (0–1, higher = more risk-off)
    updated_at: str        # ISO timestamp
    features: Dict[str, Any]


class RegimeDetector:
    """
    Phase 120 – Macro Regime Detector

    Uses:
      - VIX (vix)
      - 20d SPX return (spx_ret_20d)
      - Yield curve slope (yield_curve_10y_2y_bps)

    from a CSV file with latest row like:

        date,vix,spx_ret_20d,yield_curve_10y_2y_bps
        2025-11-28,19.5,0.012,-35.0

    If the file or fields are missing, falls back to NEUTRAL regime.
    """

    def __init__(self, cfg: Optional[RegimeDetectorConfig] = None):
        """
        Universal constructor.
    
        Supports:
        - Phase 26 runner → RegimeDetector(cfg)
        - Older versions → RegimeDetector() with defaults
        """
        # If no cfg passed, build a default config
        if cfg is None:
            cfg = RegimeDetectorConfig()
    
        self.cfg = cfg
    
        # Paths (your real detector already uses these)
        try:
            self.state_path = Path(cfg.state_path)
        except Exception:
            self.state_path = Path("data/regime/state.json")
    
        try:
            self.macro_path = Path(cfg.macro_features_path)
        except Exception:
            self.macro_path = Path("data/regime/macro.json")
    
        # Internal state placeholders
        self.regime_state = {}
        self.macro_state = {}
    
        # You may also preload or lazily load states here
        # (this preserves all existing downstream logic)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_regime(self, portfolio_snapshot: Optional[Dict[str, Any]] = None) -> RegimeState:
        """
        Main entrypoint: returns current RegimeState and writes it to state_path.

        portfolio_snapshot is currently unused but kept for future extensions
        (e.g. realized vol, drawdown).
        """
        if not self.cfg.enabled:
            logger.info("RegimeDetector: disabled via config → NEUTRAL.")
            regime = self._neutral_regime(features={"reason": "disabled"})
            self._persist_state(regime)
            return regime

        features = self._load_latest_macro_features()
        regime = self._detect_from_features(features)
        self._persist_state(regime)
        return regime

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def _load_latest_macro_features(self) -> Dict[str, Any]:
        """
        Load the last row from macro_features_path CSV.
        If the file is missing or empty, returns {}.
        """
        if not self.macro_path.exists():
            logger.warning(
                "RegimeDetector: macro features file %s not found; using NEUTRAL.",
                self.macro_path,
            )
            return {}

        last_row: Dict[str, Any] | None = None
        try:
            with self.macro_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_row = row
        except Exception:
            logger.exception(
                "RegimeDetector: failed to read macro features from %s",
                self.macro_path,
            )
            return {}

        if not last_row:
            logger.warning(
                "RegimeDetector: macro features file %s is empty; using NEUTRAL.",
                self.macro_path,
            )
            return {}

        # Normalize types
        features: Dict[str, Any] = {}
        for k, v in last_row.items():
            if v is None or v == "":
                continue
            try:
                if k in ("vix", "spx_ret_20d", "yield_curve_10y_2y_bps"):
                    features[k] = float(v)
                else:
                    features[k] = v
            except Exception:
                features[k] = v

        return features

    def _persist_state(self, regime: RegimeState) -> None:
        """
        Persist current regime to JSON so other components can read it.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(regime), f, indent=2)
        except Exception:
            logger.exception(
                "RegimeDetector: failed to write regime state to %s",
                self.state_path,
            )

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------
    def _neutral_regime(self, features: Optional[Dict[str, Any]] = None) -> RegimeState:
        return RegimeState(
            name="NEUTRAL",
            risk_level="MEDIUM",
            score=0.5,
            updated_at=datetime.utcnow().isoformat(),
            features=features or {},
        )

    def _detect_from_features(self, features: Dict[str, Any]) -> RegimeState:
        if not features:
            return self._neutral_regime(features={"reason": "no_macro_data"})

        vix = float(features.get("vix", 0.0) or 0.0)
        spx_ret = float(features.get("spx_ret_20d", 0.0) or 0.0)
        curve = float(features.get("yield_curve_10y_2y_bps", 0.0) or 0.0)

        th = self.cfg.thresholds

        # Heuristic scoring:
        risk_score = 0.0  # 0 = risk-on, 1 = max risk-off

        # VIX contribution
        if vix >= th.vix_risk_off:
            risk_score += 0.5
        elif vix <= th.vix_risk_on:
            risk_score += 0.0
        else:
            # linear between vix_risk_on and vix_risk_off
            span = max(th.vix_risk_off - th.vix_risk_on, 1e-6)
            risk_score += 0.5 * (vix - th.vix_risk_on) / span

        # SPX 20d return
        if spx_ret <= th.spx_ret_risk_off:
            risk_score += 0.3
        elif spx_ret >= th.spx_ret_risk_on:
            risk_score += 0.0
        else:
            # linear interpolation
            span = max(th.spx_ret_risk_on - th.spx_ret_risk_off, 1e-6)
            risk_score += 0.3 * (th.spx_ret_risk_on - spx_ret) / span

        # Yield curve inversion (negative = inverted)
        if curve <= th.curve_inverted_bps:
            risk_score += 0.2
        else:
            risk_score += 0.0

        # Clamp score to [0,1]
        risk_score = max(0.0, min(1.0, risk_score))

        # Map score to regime
        if risk_score <= 0.3:
            name = "RISK_ON"
            risk_level = "LOW"
        elif risk_score >= 0.7:
            name = "RISK_OFF"
            risk_level = "HIGH"
        else:
            name = "NEUTRAL"
            risk_level = "MEDIUM"

        features_out = dict(features)
        features_out["risk_score"] = risk_score

        regime = RegimeState(
            name=name,
            risk_level=risk_level,
            score=risk_score,
            updated_at=datetime.utcnow().isoformat(),
            features=features_out,
        )

        logger.info(
            "RegimeDetector: %s (risk_level=%s, score=%.3f, vix=%.2f, spx_ret_20d=%.3f, curve_bps=%.1f)",
            name,
            risk_level,
            risk_score,
            vix,
            spx_ret,
            curve,
        )
        return regime
