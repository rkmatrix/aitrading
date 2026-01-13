# ai/execution/execution_pipeline_with_recorder.py
"""
Phase 63 – Execution Pipeline + Decision Recorder Wrapper

Instead of modifying ExecutionPipeline directly (Phase 62),
we provide a thin wrapper that adds decision recording:

    wrapped = ExecutionPipelineWithRecorder(pipeline, recorder, mode="PAPER")
    result = wrapped.decide_execute_and_record(ctx)

This keeps Phase 62 unchanged and lets Phase 63 be purely additive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ai.agents.base_agent import AgentContext
from ai.analytics.decision_recorder import DecisionRecorder
from ai.execution.execution_pipeline import ExecutionPipeline, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPipelineWithRecorder:
    """
    Composition wrapper around:
        • ExecutionPipeline (Phase 62)
        • DecisionRecorder (Phase 63)
    """

    pipeline: ExecutionPipeline
    recorder: DecisionRecorder
    mode: str = "DEMO"

    def decide_execute_and_record(
        self,
        ctx: AgentContext,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Delegates decision & execution to pipeline, then records the outcome.
        """
        result = self.pipeline.decide_and_execute(ctx)

        try:
            self.recorder.record(
                ctx=ctx,
                decision=result.decision,
                result=result,
                mode=self.mode,
                extra=extra or {},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("💥 Failed to record decision: %s", exc, exc_info=True)

        return result


__all__ = ["ExecutionPipelineWithRecorder"]
