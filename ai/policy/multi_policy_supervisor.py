"""
ai/policy/multi_policy_supervisor.py
------------------------------------

Phase 77 — Multi-Policy Supervisor
Phase 84 — Regime Awareness & Volatility-Adaptive Weighting
Phase 87 — Micro-Pattern Feature Awareness
Phase 88.2 — AlphaBrain Fusion (Performance-Based Auto-Swap, Option C)
Phase 88.3 — Evolution Memory (persistent per-policy performance)

This module combines decisions from multiple RL policies based on:
    • execution-aware metrics (slippage, latency, fill-prob)
    • realized volatility & drawdown
    • intraday regime (quiet/volatile/range/etc.)
    • micro-pattern features (momentum, volume spike, breakout/squeeze)
    • per-policy performance score (Option C auto-swap)
    • persistent performance memory across runs (Phase 88.3)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3 import PPO
import yaml

from ai.regime.regime_detector import detect_intraday_regime
from ai.policy.perf_recorder import PerformanceRecorder

logger = logging.getLogger(__name__)


class PolicyBundle:
    """
    Wraps a single RL policy (PPO) + metadata + health metrics.
    """

    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.model_path = model_path
        self.model = PPO.load(str(model_path))

        # "health" = fast-moving quality indicator (from execution metrics)
        self.health_score: float = 1.0
        self.last_slippage: float = 0.0
        self.last_latency: float = 0.0
        self.last_fill_prob: float = 1.0
        self.perf_rec = PerformanceRecorder()


class MultiPolicySupervisor:
    """
    Main Supervisor.

    Combines multiple RL policies with adaptive weights based on:
        - execution metrics
        - drawdown / volatility
        - intraday regime
        - per-policy performance (Option C: performance-based auto-swap)
        - persistent performance memory (Phase 88.3)
    """

    # For AlphaBrain obs reconstruction (must match AlphaBrainObsWrapper)
    FEATURE_KEYS = [
        "mom_1",
        "mom_5",
        "mom_10",
        "vol_20",
        "vol_60",
        "vol_ratio_20_60",
        "volume_ratio",
        "intraday_range_norm",
        "price_pos_in_range",
        "breakout_score",
        "squeeze_score",
    ]

    REGIME_KEYS = [
        "quiet_trend",
        "rangebound",
        "volatile_trend",
        "chaos",
        "extreme_vol",
    ]

    def __init__(self, config_path: str = "configs/phase77_supervisor.yaml"):
        self.cfg = self._load_cfg(config_path)

        self.policies: Dict[str, PolicyBundle] = {}
        self._load_policy_bundles()

        modes = self.cfg.get("modes", {})
        weights = self.cfg.get("weights", {})
        paths = self.cfg.get("paths", {})

        self.volatility_threshold = float(modes.get("volatility_threshold", 0.02))
        self.safety_drawdown = float(modes.get("drawdown_limit", 0.10))

        self.execution_penalty_weight = float(weights.get("execution_penalty", 0.5))
        self.performance_boost_weight = float(weights.get("performance_boost", 0.5))
        # how much we trust performance score vs health score when building weights
        self.perf_weight = float(weights.get("perf_weight", 0.7))
        self.health_weight = float(weights.get("health_weight", 0.3))

        self.num_actions = int(self.cfg.get("num_actions", 3))

        # --- Phase 88.3: persistent evolution memory path ---
        self.perf_state_path = Path(
            paths.get("perf_state_path", "data/reports/policy_perf_state.json")
        )

        # --- Option C: per-policy performance state ---
        # name -> dict(pnl_ema, slip_ema, lat_ema, fill_ema, score, count)
        self.policy_perf: Dict[str, Dict[str, float]] = {}
        self.last_chosen_policy: Optional[str] = None

        self._perf_state_loaded = False
        self._load_perf_state()

        logger.info(
            "Phase 77/84/87/88/88.3 MultiPolicySupervisor initialized with %d policies",
            len(self.policies),
        )

    # -------------------------------------------------------
    # Config / loading
    # -------------------------------------------------------

    def _load_cfg(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_policy_bundles(self) -> None:
        base_dir = Path(self.cfg.get("paths", {}).get("policy_root", "models/policies"))
        active = self.cfg.get("active_policies", [])

        for pol in active:
            mod = base_dir / pol / "model.zip"
            if mod.exists():
                self.policies[pol] = PolicyBundle(pol, mod)
                logger.info("Loaded policy '%s' from %s", pol, mod)
            else:
                logger.warning("Policy '%s' not found at %s", pol, mod)

    # -------------------------------------------------------
    # Persistent evolution memory (Phase 88.3)
    # -------------------------------------------------------

    def _load_perf_state(self) -> None:
        """
        Load policy performance state from disk (if available).
        """
        if self._perf_state_loaded:
            return

        self._perf_state_loaded = True
        if not self.perf_state_path.exists():
            logger.info(
                "MultiPolicySupervisor: No existing perf_state at %s (fresh start).",
                self.perf_state_path,
            )
            return

        try:
            with open(self.perf_state_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(
                "MultiPolicySupervisor: Failed to load perf_state from %s: %s",
                self.perf_state_path,
                e,
            )
            return

        if not isinstance(data, dict):
            logger.warning(
                "MultiPolicySupervisor: perf_state file %s has invalid format.",
                self.perf_state_path,
            )
            return

        for name, st in data.items():
            if not isinstance(st, dict):
                continue
            self.policy_perf[name] = {
                "pnl_ema": float(st.get("pnl_ema", 0.0)),
                "slip_ema": float(st.get("slip_ema", 0.0)),
                "lat_ema": float(st.get("lat_ema", 0.0)),
                "fill_ema": float(st.get("fill_ema", 1.0)),
                "score": float(st.get("score", 0.0)),
                "count": float(st.get("count", 0.0)),
            }

        logger.info(
            "MultiPolicySupervisor: Loaded evolution memory for %d policies from %s",
            len(self.policy_perf),
            self.perf_state_path,
        )

    def _save_perf_state(self) -> None:
        """
        Save policy performance state to disk.
        """
        try:
            self.perf_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.perf_state_path, "w") as f:
                json.dump(self.policy_perf, f, indent=2)
        except Exception as e:
            logger.warning(
                "MultiPolicySupervisor: Failed to save perf_state to %s: %s",
                self.perf_state_path,
                e,
            )

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------

    def _ensure_perf_state(self) -> None:
        for name in self.policies.keys():
            if name not in self.policy_perf:
                self.policy_perf[name] = {
                    "pnl_ema": 0.0,
                    "slip_ema": 0.0,
                    "lat_ema": 0.0,
                    "fill_ema": 1.0,
                    "score": 0.0,
                    "count": 0.0,
                }

    def _update_health_from_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Fast-moving "health" update shared across policies
        (slippage, latency, fill, pnl). This does NOT differentiate between
        policies; it just captures current market conditions / execution quality.
        """
        for p in self.policies.values():
            slip = float(metrics.get("slippage", 0.0))
            lat = float(metrics.get("latency", 0.0))
            fill = float(metrics.get("fill_prob", 1.0))
            pnl = float(metrics.get("pnl", 0.0))

            exec_penalty = abs(slip) + max(0.0, lat / 3000.0) - fill
            perf_boost = max(0.0, pnl)

            score = (
                1.0
                - self.execution_penalty_weight * exec_penalty
                + self.performance_boost_weight * perf_boost
            )

            p.health_score = max(0.05, score)
            p.last_slippage = slip
            p.last_latency = lat
            p.last_fill_prob = fill

    def _update_perf_from_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Option C — performance-based auto-swap + Phase 88.3 persistence.

        We attribute the latest metrics to the policy that was chosen
        on the previous step (self.last_chosen_policy). Over time,
        each policy gets a performance score "score" that reflects:
            pnl_ema - |slip_ema| - latency_penalty + fill_ema

        After updating, we persist to disk.
        """
        self._ensure_perf_state()
        name = self.last_chosen_policy
        if not name or name not in self.policy_perf:
            return

        st = self.policy_perf[name]

        pnl = float(metrics.get("pnl", 0.0))
        slip = float(metrics.get("slippage", 0.0))
        lat = float(metrics.get("latency", 0.0))
        fill = float(metrics.get("fill_prob", 1.0))

        alpha = 0.1  # EMA smoothing

        st["pnl_ema"] = (1 - alpha) * st["pnl_ema"] + alpha * pnl
        st["slip_ema"] = (1 - alpha) * st["slip_ema"] + alpha * slip
        st["lat_ema"] = (1 - alpha) * st["lat_ema"] + alpha * lat
        st["fill_ema"] = (1 - alpha) * st["fill_ema"] + alpha * fill
        st["count"] += 1.0

        perf_score = (
            st["pnl_ema"]
            - abs(st["slip_ema"])
            - max(0.0, st["lat_ema"] / 3000.0)
            + st["fill_ema"]
        )
        st["score"] = perf_score

        # Persist evolution memory
        self._save_perf_state()

    # -------------------------------------------------------
    # Weight logic (base + regime + perf)
    # -------------------------------------------------------

    def _compute_base_weights(self, volatility: float, drawdown: float) -> Dict[str, float]:
        """
        Base weights from volatility, drawdown and policy health.
        """
        self._ensure_perf_state()

        # Safety mode: large drawdown → prefer safer / more execution-aware policies
        if drawdown >= self.safety_drawdown:
            weights = {}
            for name, p in self.policies.items():
                if "ExecAware" in name:
                    weights[name] = p.health_score * 1.5
                else:
                    weights[name] = p.health_score * 0.5
            return weights

        # High-volatility mode
        if volatility >= self.volatility_threshold:
            weights = {}
            for name, p in self.policies.items():
                if "ExecAware" in name:
                    weights[name] = p.health_score * 1.2
                else:
                    weights[name] = p.health_score * 0.8
            return weights

        # Default: proportional to health_score
        return {name: p.health_score for name, p in self.policies.items()}

    def _apply_regime_override(
        self,
        prices: Optional[np.ndarray],
        base_weights: Dict[str, float],
        volatility: float,
        drawdown: float,
    ) -> Dict[str, float]:
        """
        Regime-aware reweighting (Phase 84).

        For now we only log regime and keep minor adjustments;
        future versions can strongly reweight per policy by regime.
        """
        if prices is None:
            return base_weights

        try:
            regime = detect_intraday_regime(prices)
        except Exception:
            regime = "unknown"

        logger.debug(
            "MultiPolicySupervisor: regime=%s vol=%.6f dd=%.6f",
            regime,
            volatility,
            drawdown,
        )

        # Example: in extreme_vol, tilt a bit more towards ExecAware
        weights = dict(base_weights)
        if regime == "extreme_vol":
            for name in weights:
                if "ExecAware" in name:
                    weights[name] *= 1.2
                else:
                    weights[name] *= 0.8

        return weights

    def _apply_perf_override(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Option C — performance-based auto-swap:

        Combine health_score and perf_score to derive new weights.
        """
        self._ensure_perf_state()
        new_weights: Dict[str, float] = {}

        for name, base_w in weights.items():
            perf = self.policy_perf.get(name, {}).get("score", 0.0)
            health = self.policies[name].health_score

            combined = (
                self.health_weight * health +
                self.perf_weight * perf
            )

            # Keep at least tiny positive weight for exploration
            new_weights[name] = max(0.01, combined)

        return new_weights

    # -------------------------------------------------------
    # AlphaBrain obs reconstruction (live side)
    # -------------------------------------------------------

    def _build_alpha_obs(
        self,
        window: np.ndarray,
        metrics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Build the same style of feature vector AlphaBrain was trained on:
            [flattened_window, micro_features..., volatility, regime_onehot...]

        We reuse:
            - metrics["micro_features"] (from ExecutionAwareLiveAgent)
            - metrics["volatility"]
            - detect_intraday_regime(window)
        """
        if window is None or window.ndim != 2:
            return np.zeros(1, dtype=np.float32)

        flat_win = window.astype(np.float32).flatten()

        micro = metrics.get("micro_features") or {}
        micro_vec = np.array(
            [float(micro.get(k, 0.0)) for k in self.FEATURE_KEYS],
            dtype=np.float32,
        )

        vol = float(metrics.get("volatility", 0.0))
        vol_vec = np.array([vol], dtype=np.float32)

        try:
            regime = detect_intraday_regime(window)
        except Exception:
            regime = "unknown"

        regime_onehot = np.zeros(len(self.REGIME_KEYS), dtype=np.float32)
        if regime in self.REGIME_KEYS:
            idx = self.REGIME_KEYS.index(regime)
            regime_onehot[idx] = 1.0

        extra = np.concatenate([micro_vec, vol_vec, regime_onehot]).astype(np.float32)
        return np.concatenate([flat_win, extra], dtype=np.float32)

    # -------------------------------------------------------
    # Public: choose_action
    # -------------------------------------------------------

    def choose_action(
        self,
        obs: np.ndarray,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main decision function.

        obs:
            - expected to be the raw window from ExecutionAwareLiveAgent,
              shape (window, 5).
        metrics:
            - dict including volatility, drawdown, micro_features, etc.
        """
        if not self.policies:
            raise RuntimeError("MultiPolicySupervisor: no policies loaded")

        if obs is None:
            raise ValueError("MultiPolicySupervisor.choose_action: obs is None")

        if obs.ndim != 2:
            raise ValueError(
                f"MultiPolicySupervisor.choose_action expects obs shape (window, features), got {obs.shape}"
            )

        volatility = float(metrics.get("volatility", 0.0))
        drawdown = float(metrics.get("drawdown", 0.0))
        prices = metrics.get("recent_prices", None)

        # 1) Update health & performance from latest metrics
        self._update_health_from_metrics(metrics)
        self._update_perf_from_metrics(metrics)

        # 2) Base weights (health + vol / drawdown)
        weights = self._compute_base_weights(volatility, drawdown)

        # 3) Regime-aware tilt
        weights = self._apply_regime_override(
            prices=prices,
            base_weights=weights,
            volatility=volatility,
            drawdown=drawdown,
        )

        # 4) Performance-based override (Option C + 88.3 memory)
        weights = self._apply_perf_override(weights)

        # 5) Normalize weights
        total = sum(weights.values())
        if total <= 0:
            weights = {k: 1.0 / len(weights) for k in weights}
        else:
            for k in list(weights.keys()):
                weights[k] = float(weights[k] / total)

        # ---------------------------------------------------
        # Get actions from each policy with proper obs shape
        # ---------------------------------------------------
        breakdown: Dict[str, float] = {}
        blend = np.zeros(self.num_actions, dtype=np.float32)

        for name, bundle in self.policies.items():
            if name == "AlphaBrainPolicy":
                alpha_obs = self._build_alpha_obs(obs, metrics)
                obs_batch = alpha_obs.reshape(1, -1)
            else:
                # Exec-aware style policies trained on TradingEnv-v0 (Box(60,5))
                obs_batch = obs.reshape(1, *obs.shape)  # (1, window, 5)

            try:
                action_raw, _ = bundle.model.predict(obs_batch, deterministic=False)
            except Exception as e:
                logger.error(
                    "MultiPolicySupervisor: predict failed for policy '%s': %s",
                    name,
                    e,
                )
                continue

            # Convert to discrete distribution / one-hot
            if np.isscalar(action_raw):
                idx = int(action_raw)
                if 0 <= idx < self.num_actions:
                    one_hot = np.zeros(self.num_actions, dtype=np.float32)
                    one_hot[idx] = 1.0
                else:
                    one_hot = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
            else:
                arr = np.array(action_raw).flatten()
                if arr.size == 1:
                    idx = int(arr[0])
                    one_hot = np.zeros(self.num_actions, dtype=np.float32)
                    if 0 <= idx < self.num_actions:
                        one_hot[idx] = 1.0
                    else:
                        one_hot[:] = 1.0 / self.num_actions
                else:
                    # Assume already a discrete-prob distribution or logits; softmax-ish
                    exps = np.exp(arr - np.max(arr))
                    prob = exps / np.sum(exps)
                    if prob.size != self.num_actions:
                        # Pad / trim to num_actions
                        if prob.size < self.num_actions:
                            prob = np.pad(prob, (0, self.num_actions - prob.size))
                        else:
                            prob = prob[: self.num_actions]
                    one_hot = prob.astype(np.float32)

            w = float(weights.get(name, 0.0))
            blend += one_hot * w
            breakdown[name] = w

        # If for some reason blend is all zeros, fallback to uniform
        if np.allclose(blend, 0.0):
            blend[:] = 1.0 / self.num_actions

        final_action = int(np.argmax(blend))

        # The policy with highest weight (for interpretability)
        chosen_policy = max(weights, key=lambda k: weights[k]) if weights else None
        self.last_chosen_policy = chosen_policy

        # ---------------------------------------------------------------
        # Phase 88.3 PerformanceRecorder integration
        # ---------------------------------------------------------------
        # Record a "synthetic reward" based on policy performance.
        # This does not use PnL, because reward is generated inside Phase 26.
        # Here we log only policy selection events.
        try:
            if chosen_policy:
                # Policy is considered to have "positive" outcome when chosen,
                # negative when down-weighted; simple but effective signal.
                reward = float(weights.get(chosen_policy, 0.0))
                win_flag = 1 if reward > 0 else 0
                self.policies[chosen_policy].perf_rec.record(
                    policy_name=chosen_policy,
                    reward=reward,
                    win_flag=win_flag,
                )
        except Exception as e:
            logger.error(f"Perf logging error in supervisor: {e}")
        # ---------------------------------------------------------------

        return {
            "action": final_action,
            "policy_breakdown": breakdown,
            "chosen_policy": chosen_policy,
            "weights": weights,
        }

    def reload(self, config_path: str | None = None) -> bool:
        """
        Safe in-place reload:
        - Re-reads config (optional new path)
        - Re-loads active policy bundles into a candidate dict
        - Swaps policies only if at least 1 policy loads successfully
        Returns True if swap happened, else False.
        """
        try:
            if config_path:
                self.cfg = self._load_cfg(config_path)
    
            base_dir = Path(self.cfg.get("paths", {}).get("policy_root", "models/policies"))
            active = self.cfg.get("active_policies", [])
    
            candidate: Dict[str, PolicyBundle] = {}
            for pol in active:
                mod = base_dir / pol / "model.zip"
                if not mod.exists():
                    logger.warning("HotReload: Policy '%s' not found at %s", pol, mod)
                    continue
                try:
                    candidate[pol] = PolicyBundle(pol, mod)
                    logger.info("HotReload: loaded policy '%s' from %s", pol, mod)
                except Exception as e:
                    logger.error("HotReload: failed loading policy '%s': %s", pol, e, exc_info=True)
    
            if not candidate:
                logger.error("HotReload: no policies loaded; keeping existing policies.")
                return False
    
            # Keep existing perf memory; refresh from disk if present
            # (doesn't wipe state; just ensures new policies can use it)
            self._perf_state_loaded = False
            self._ensure_perf_state()
    
            self.policies = candidate
            logger.warning("✅ HotReload: swapped in %d policies.", len(self.policies))
            return True
    
        except Exception as e:
            logger.error("HotReload: reload failed; keeping existing policies. err=%s", e, exc_info=True)
            return False
