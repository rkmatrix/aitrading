import numpy as np
import logging
from ai.env.volatility_meter import VolatilityMeter

logger = logging.getLogger(__name__)

class VolatilityThrottler:
    """
    Scales scalar or vector actions using aggregate portfolio volatility.
    - Uses position-weighted aggregate if configured + positions provided in info.
    - Falls back to equal-weight aggregate or single-price path.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.vol_meter = VolatilityMeter(cfg)
        risk_cfg = cfg["risk_aware"]
        self.thresholds = risk_cfg["vol_thresholds"]
        self.factors = risk_cfg["vol_throttle"]
        self.symbols = cfg.get("symbols", [])
        self._last_agg_vol = 0.0

    def update(self, info):
        prices = info.get("prices")
        pos_dollars = info.get("positions_dollars")  # dict symbol->$exposure
        if isinstance(prices, dict) and prices:
            vols = self.vol_meter.update_dict(prices, pos_dollars)
            self._last_agg_vol = vols["aggregate"]
        else:
            price = info.get("price")
            if price is not None:
                self._last_agg_vol = self.vol_meter.update(price)

    def _pick_factor(self, vol):
        if vol < self.thresholds["low"]:
            return self.factors["low_vol"]
        elif vol > self.thresholds["high"]:
            return self.factors["high_vol"]
        return self.factors["mid_vol"]

    def scale_action(self, action):
        vol = self._last_agg_vol or self.vol_meter.current_vol()
        factor = self._pick_factor(vol)

        # Support scalar or vector actions
        action_np = np.array(action, dtype=np.float64)
        scaled = np.clip(action_np * factor, -1.0, 1.0)

        logger.debug(f"🎚️ agg_vol={vol:.5f}, factor={factor:.3f}, scaled={scaled}")
        # keep original type/shape fidelity
        if np.isscalar(action):
            return float(scaled.item())
        return scaled
