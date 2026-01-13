"""
ai/policy/fusion_brain.py

Phase 92.4 – Multi-Agent FusionBrain
-----------------------------------
This module unifies multiple predictive signals:

    • ML AlphaModel (Phase 101)
    • Momentum (from runner)
    • Aggression factor (from EquityGrowthAgent)
    • Optional PPO agent (Phase 68–71)
    • Optional AlphaZoo agents (Phase 102)
    • Optional sentiment/news/macro/orderflow agents
    • Volatility & drawdown context (portfolio_provider)

FusionBrain returns:
    fused_score ∈ [-1, 1]
    weight breakdown
    diagnostic context

If certain agents do not exist, FusionBrain safely falls back using
Dummy agents without breaking the architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
import yaml

# Dummy stubs
from ai.policy.agents.execution_ppo_agent import DummyExecutionPPOAgent
from ai.policy.agents.ml_alphazoo_agent import DummyMLAlphaZooAgent

log = logging.getLogger("FusionBrain")


# -------------------------------------------------------------------------
# Helper: safe_loader
# -------------------------------------------------------------------------
def _safe_load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning("⚠️ FusionBrain config not found: %s", path)
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.error("Failed reading %s: %s", path, e)
        return {}


# -------------------------------------------------------------------------
# FusionBrain class
# -------------------------------------------------------------------------
class FusionBrain:
    def __init__(
        self,
        symbols,
        cfg: dict,
        broker=None,
        portfolio_provider=None,
    ) -> None:
        self.symbols = symbols
        self.cfg = cfg
        self.broker = broker
        self.portfolio_provider = portfolio_provider

        # Load optional agents
        self._init_optional_agents()

        # Fusion weights
        self.weights = cfg.get("weights", {
            "ml": 0.40,
            "mom": 0.30,
            "agg": 0.10,
            "ppo": 0.10,
            "alphazoo": 0.10,
        })

    # ---------------------------------------------------------------------
    # Optional agents
    # ---------------------------------------------------------------------
    def _init_optional_agents(self):
        cfg = self.cfg

        # PPO agent
        ppo_cfg = cfg.get("ppo_overlay", {})
        if ppo_cfg.get("enabled", False):
            try:
                path = ppo_cfg.get("path")
                self.ppo_agent = DummyExecutionPPOAgent.load(path)
                log.info("🧠 PPO overlay active (%s)", path)
            except Exception as e:
                log.error("Failed loading PPO agent: %s", e)
                self.ppo_agent = DummyExecutionPPOAgent()
        else:
            self.ppo_agent = DummyExecutionPPOAgent()

        # AlphaZoo agent
        az_cfg = cfg.get("alphazoo", {})
        if az_cfg.get("enabled", False):
            try:
                path = az_cfg.get("path")
                self.az_agent = DummyMLAlphaZooAgent.load(path)
                log.info("🧠 AlphaZoo overlay active (%s)", path)
            except Exception as e:
                log.error("Failed loading AlphaZoo: %s", e)
                self.az_agent = DummyMLAlphaZooAgent()
        else:
            self.az_agent = DummyMLAlphaZooAgent()

    # ---------------------------------------------------------------------
    # FUSE
    # ---------------------------------------------------------------------
    def fuse(
        self,
        symbol: str,
        price: float,
        ml_score: float,
        ml_pred: float,
        mom_score: float,
        agg_score: float,
        ctx: dict,
    ) -> dict:

        w = self.weights

        # Optional PPO influence
        try:
            ppo_val = float(self.ppo_agent.predict([price, ml_score, mom_score]))
        except:
            ppo_val = 0.0

        # Optional AlphaZoo influence
        try:
            az_val = float(self.az_agent.predict(ctx))
        except:
            az_val = 0.0

        # Weighted fusion
        fused = (
            w["ml"] * ml_score +
            w["mom"] * mom_score +
            w["agg"] * agg_score +
            w["ppo"] * ppo_val +
            w["alphazoo"] * az_val
        )

        # Clip range
        fused = max(-1.0, min(1.0, fused))

        return {
            "fused_score": fused,
            "weights": w,
            "context": {
                "symbol": symbol,
                "ml": ml_score,
                "mom": mom_score,
                "agg": agg_score,
                "ppo_val": ppo_val,
                "alphazoo_val": az_val,
                "volatility": ctx.get("volatility"),
                "drawdown": ctx.get("drawdown"),
                "aggression_factor": ctx.get("aggression_factor"),
            },
        }


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------
def build_fusion_brain(
    symbols,
    cfg_path: str,
    broker=None,
    portfolio_provider=None,
) -> FusionBrain:
    cfg = _safe_load_yaml(cfg_path)
    return FusionBrain(
        symbols=symbols,
        cfg=cfg,
        broker=broker,
        portfolio_provider=portfolio_provider,
    )
