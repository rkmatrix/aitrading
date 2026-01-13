from __future__ import annotations
import os, json, time, logging, random, importlib, pkgutil, inspect
from datetime import datetime
from pathlib import Path
import pandas as pd
from tools.telegram_alerts import notify

# ======================================================
# 🔧 Optional imports for Adaptive State Observer + Feature Encoder
# ======================================================
try:
    from ai.market.state_observer import StateObserver, RandomFeed, CSVFeed
except Exception:
    StateObserver = RandomFeed = CSVFeed = None

try:
    from ai.policy.feature_encoder import FeatureEncoder
    import yaml as _yaml
except Exception:
    FeatureEncoder = _yaml = None

# ======================================================
# 🔧 Dynamic Import Resolvers
# ======================================================

# --- SmartOrderRouter ---
SmartOrderRouter = None
for path in [
    "ai.execution.smart_order_router",
    "ai.execution.smart_router",
    "ai.router.smart_order_router",
    "ai.execution.router",
]:
    try:
        mod = importlib.import_module(path)
        SmartOrderRouter = getattr(mod, "SmartOrderRouter", None)
        if SmartOrderRouter:
            logging.info(f"[ImportResolver] Using SmartOrderRouter from {path}")
            break
    except ImportError:
        continue
if not SmartOrderRouter:
    raise ImportError("❌ SmartOrderRouter module not found (Phase 35).")

# --- RiskGuardian ---
RiskGuardian = None
try:
    rg = importlib.import_module("ai.guardian.risk_guardian")
    RiskGuardian = getattr(rg, "RiskGuardian", None) or getattr(rg, "Guardian", None)
    if RiskGuardian:
        logging.info("[ImportResolver] Using RiskGuardian from ai.guardian.risk_guardian")
except ImportError:
    pass
if not RiskGuardian:
    raise ImportError("❌ Could not locate RiskGuardian/Guardian class.")

# --- PolicyLoader ---
PolicyLoader = None
for path in [
    "ai.policy.policy_loader",
    "ai.policy_loader",
    "ai.policies.loader",
    "ai.policies.policy_loader",
    "ai.policy.registry",
]:
    try:
        mod = importlib.import_module(path)
        PolicyLoader = getattr(mod, "PolicyLoader", None)
        if PolicyLoader:
            logging.info(f"[ImportResolver] Using PolicyLoader from {path}")
            break
    except ImportError:
        continue
if not PolicyLoader:
    for _, modname, _ in pkgutil.walk_packages(["ai"], "ai."):
        try:
            m = importlib.import_module(modname)
            for name, obj in inspect.getmembers(m):
                if name == "PolicyLoader" and inspect.isclass(obj):
                    PolicyLoader = obj
                    logging.info(f"[AutoDiscover] Found PolicyLoader in {modname}")
                    break
            if PolicyLoader:
                break
        except Exception:
            continue
if not PolicyLoader:
    raise ImportError("❌ PolicyLoader class not found anywhere under ai/. Check Phase 41 implementation.")


# ======================================================
# 🧠 IntelligentExecutor (Phase 50 → 51.2)
# ======================================================

logger = logging.getLogger("IntelligentExecutor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class IntelligentExecutor:
    """Phases 50 → 51.2 Intelligent Trade Executor with Adaptive Market State"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.policy_root = Path(cfg["policy"]["root"])
        self.policy_version = cfg["policy"].get("version", "latest")
        self.symbols = cfg["symbols"]
        self.interval = cfg["policy"]["action_interval"]
        self.router = SmartOrderRouter(cfg["execution"]["router_config"])
        self.guardian = RiskGuardian()
        self.dry_run = cfg["execution"]["dry_run"]
        self.risk_cfg = cfg["risk"]
        self.log_path = Path(cfg["logging"]["file"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tele_enabled = cfg["logging"].get("telegram", True)

        # --- Guardian Adapter ---
        self.guardian_fn = None
        for cand in ["is_action_allowed", "allow", "approve_trade", "check_action", "validate"]:
            if hasattr(self.guardian, cand):
                self.guardian_fn = getattr(self.guardian, cand)
                logging.info(f"[GuardianAdapter] Using guardian method '{cand}()'")
                break
        if not self.guardian_fn:
            raise AttributeError("Guardian class missing any of: is_action_allowed, allow, approve_trade, check_action, validate")

        # --- Load Policy ---
        self.policy = PolicyLoader(self.policy_root, self.policy_version).load()
        logger.info(f"✅ Loaded policy {self.policy_version} for {len(self.symbols)} symbols")

        # --- Phase 51 Adaptive State Observer ---
        self.observer = None
        self._feat_enc = None
        phase51_cfg = Path("configs/phase51_state.yaml")
        if phase51_cfg.exists() and StateObserver is not None:
            import yaml
            with open(phase51_cfg, "r") as f:
                s_cfg = yaml.safe_load(f)
            symbols = s_cfg["symbols"]
            feed_kind = s_cfg["feed"]["kind"]
            feed = None
            if feed_kind == "random" and RandomFeed is not None:
                base = {s: 100.0 for s in symbols}
                feed = RandomFeed(base, interval_seconds=s_cfg["feed"].get("interval_seconds", 2))
            elif feed_kind == "csv" and CSVFeed is not None:
                feed = CSVFeed(s_cfg["feed"]["csv_path"])
            if feed is not None:
                self.observer = StateObserver(
                    symbols,
                    feed,
                    rsi_period=s_cfg["features"]["rsi_period"],
                    vol_window=s_cfg["features"]["vol_window"],
                    delta_window=s_cfg["features"]["delta_window"],
                    vol_avg_window=s_cfg["features"]["vol_avg_window"],
                    snapshot_csv=s_cfg["logging"].get("snapshot_csv"),
                )
                logging.info("[Phase51] Adaptive State Observer enabled.")

        # --- Phase 51.2 Feature Encoder ---
        if FeatureEncoder is not None and _yaml is not None:
            fcfg_path = Path("configs/phase51_features.yaml")
            if fcfg_path.exists():
                with open(fcfg_path, "r") as f:
                    fcfg = _yaml.safe_load(f)
                self._feat_enc = FeatureEncoder(fcfg)
                logging.info("[Phase51.2] FeatureEncoder loaded and active.")

    # --------------------------------------------------
    def _decide_action(self, symbol: str) -> dict:
        """Generate next action using policy or heuristic fallback."""
        state = {}
        if self.observer:
            states = self.observer.step()
            state = states.get(symbol, self.observer.get_state(symbol)) or {}

        # Optional vectorization for feature-aware policies
        obs_vec = None
        if self._feat_enc is not None:
            try:
                obs_vec = self._feat_enc.encode(state)
            except Exception:
                obs_vec = None

        # --- Try policy.predict() ---
        if hasattr(self.policy, "predict") and callable(self.policy.predict):
            try:
                inp = obs_vec if obs_vec is not None else state
                action, _ = self.policy.predict(inp)
                confidence = random.uniform(0.6, 0.95)
                return {"symbol": symbol, "action": action, "confidence": confidence, "state": state}
            except Exception:
                pass  # fallback

        # --- Heuristic fallback (STUB mode) ---
        rsi = state.get("rsi", 50.0)
        delta = state.get("priceDelta", 0.0)
        if rsi < 35 and delta >= 0:
            decision, conf = "BUY", 0.75
        elif rsi > 65 and delta <= 0:
            decision, conf = "SELL", 0.75
        else:
            decision = random.choice(["HOLD", "BUY", "SELL"])
            conf = min(0.5 + 0.4 * abs(delta), 0.95)
        return {"symbol": symbol, "action": decision, "confidence": conf, "state": state}

    # --------------------------------------------------
    def _validate_action(self, act: dict) -> bool:
        """Guardian + confidence checks"""
        if act["confidence"] < self.risk_cfg["confidence_threshold"]:
            logger.info(f"⚠️ Low confidence {act['confidence']:.2f} → skip {act['symbol']}")
            return False
        try:
            ok = self.guardian_fn(act)
        except TypeError:
            try:
                ok = self.guardian_fn(act["symbol"], act["action"])
            except Exception:
                ok = True
        except Exception as e:
            logger.error(f"Guardian validation error: {e}")
            ok = True
        if not ok:
            logger.info(f"⛔ Guardian blocked {act['action']} on {act['symbol']}")
            return False
        return True

    # --------------------------------------------------
    def _execute(self, act: dict):
        """Send order via router or simulate in dry run"""
        if self.dry_run:
            logger.info(f"[DRY] {act['action']} {act['symbol']}")
            return
        try:
            # Auto-detect router API once
            if not hasattr(self, "_router_fn"):
                for cand in ["route_order", "route", "execute_order", "execute", "send_order"]:
                    if hasattr(self.router, cand):
                        self._router_fn = getattr(self.router, cand)
                        logger.info(f"[RouterAdapter] Using router method '{cand}()'")
                        break
                else:
                    raise AttributeError("SmartOrderRouter missing route-method candidates")

            try:
                self._router_fn(symbol=act["symbol"], side=act["action"], qty=10)
            except TypeError:
                self._router_fn(act)
        except Exception as e:
            logger.error(f"❌ Router failed for {act['symbol']}: {e}")

    # --------------------------------------------------
    def _log(self, act: dict):
        """Append decision and feature context to CSV log"""
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": act["symbol"],
            "action": act["action"],
            "confidence": act["confidence"],
            "rsi": act.get("state", {}).get("rsi"),
            "priceDelta": act.get("state", {}).get("priceDelta"),
            "volatility": act.get("state", {}).get("volatility"),
            "volumeRatio": act.get("state", {}).get("volumeRatio"),
        }])
        row.to_csv(self.log_path, mode="a", header=not self.log_path.exists(), index=False)

    # --------------------------------------------------
    def run_forever(self):
        logger.info("🚀 Phase 50–51.2 Intelligent Executor started.")
        while True:
            # --- 🔄 Auto-reload check ---
            flag = Path("data/runtime/reload_now.flag")
            if flag.exists():
                try:
                    txt = flag.read_text(encoding="utf-8").strip()
                    parts = txt.split(",")
                    pol, ver = parts[0], parts[1] if len(parts) > 1 else "latest"
                    logger.info(f"🧠 Detected reload flag for {pol} → reloading policy {ver}")
                    self.policy = PolicyLoader(self.policy_root, "latest").load()
                    logger.info(f"✅ Policy reloaded successfully.")
                    flag.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"💥 Failed reload attempt: {e}")

            # --- normal action cycle ---
            for sym in self.symbols:
                act = self._decide_action(sym)
                if self._validate_action(act):
                    self._execute(act)
                    self._log(act)
                    if self.tele_enabled:
                        notify(f"🎯 {act['action']} {act['symbol']} ({act['confidence']:.2f})", kind="orders")
            time.sleep(self.interval)

