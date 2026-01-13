from __future__ import annotations
import json, logging, os, socket, time, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubsystemHealth:
    name: str
    status: str  # "ok" | "fail"
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] | None = None


class SubsystemPinger:
    """
    Lightweight health checks for external dependencies.

    - Alpaca REST (if credentials are present)
    - Generic internet connectivity
    - Filesystem writeability
    - Policy bundle presence
    - PortfolioBrain presence (if importable)
    """

    def __init__(
        self,
        *,
        policy_dir: str | Path = "models/policies",
        health_log_path: str | Path = "data/runtime/phase68_health.json",
    ) -> None:
        self.policy_dir = Path(policy_dir)
        self.health_log_path = Path(health_log_path)
        self.health_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public API -----------------------------------------------------

    def run_all(self) -> List[SubsystemHealth]:
        checks = [
            self.ping_network,
            self.ping_filesystem,
            self.ping_alpaca_rest,
            self.ping_policy_bundle,
            self.ping_portfolio_brain,
        ]
        results: List[SubsystemHealth] = []
        for fn in checks:
            try:
                res = fn()
                if res is not None:
                    results.append(res)
            except Exception as e:  # defensive
                logger.exception("Health check %s exploded", fn.__name__)
                results.append(
                    SubsystemHealth(
                        name=fn.__name__,
                        status="fail",
                        error=str(e),
                    )
                )

        # Persist snapshot for dashboards
        try:
            with self.health_log_path.open("w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
        except Exception:
            logger.exception("Failed to write health snapshot to %s", self.health_log_path)

        return results

    # ---- individual checks ---------------------------------------------

    def ping_network(self) -> SubsystemHealth:
        start = time.time()
        host = "8.8.8.8"
        port = 53
        try:
            with socket.create_connection((host, port), timeout=3):
                pass
            latency_ms = (time.time() - start) * 1000
            return SubsystemHealth(name="network", status="ok", latency_ms=latency_ms)
        except Exception as e:
            return SubsystemHealth(name="network", status="fail", error=str(e))

    def ping_filesystem(self) -> SubsystemHealth:
        start = time.time()
        test_path = self.health_log_path.parent / ".fs_probe"
        try:
            test_path.write_text("ok")
            test_path.unlink(missing_ok=True)
            latency_ms = (time.time() - start) * 1000
            return SubsystemHealth(name="filesystem", status="ok", latency_ms=latency_ms)
        except Exception as e:
            return SubsystemHealth(name="filesystem", status="fail", error=str(e))

    def ping_alpaca_rest(self) -> Optional[SubsystemHealth]:
        key_id = os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        if not key_id or not secret:
            # silently skip when not configured
            return None

        url = base_url.rstrip("/") + "/v2/account"
        start = time.time()
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("APCA-API-KEY-ID", key_id)
            req.add_header("APCA-API-SECRET-KEY", secret)
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.getcode()
                body = resp.read(256)
            latency_ms = (time.time() - start) * 1000
            if 200 <= code < 300:
                return SubsystemHealth(
                    name="alpaca_rest",
                    status="ok",
                    latency_ms=latency_ms,
                    meta={"code": code},
                )
            return SubsystemHealth(
                name="alpaca_rest",
                status="fail",
                latency_ms=latency_ms,
                error=f"HTTP {code}: {body[:100]!r}",
            )
        except Exception as e:
            return SubsystemHealth(
                name="alpaca_rest",
                status="fail",
                error=str(e),
            )

    def ping_policy_bundle(self) -> SubsystemHealth:
        start = time.time()
        try:
            if not self.policy_dir.exists():
                return SubsystemHealth(
                    name="policy_bundle",
                    status="fail",
                    error=f"Missing directory: {self.policy_dir}",
                )
            manifests = list(self.policy_dir.glob("*/manifest.json"))
            latency_ms = (time.time() - start) * 1000
            if not manifests:
                return SubsystemHealth(
                    name="policy_bundle",
                    status="fail",
                    latency_ms=latency_ms,
                    error="No manifest.json found under models/policies/*",
                )
            return SubsystemHealth(
                name="policy_bundle",
                status="ok",
                latency_ms=latency_ms,
                meta={"bundles": [str(m.parent.name) for m in manifests]},
            )
        except Exception as e:
            return SubsystemHealth(name="policy_bundle", status="fail", error=str(e))

    def ping_portfolio_brain(self) -> Optional[SubsystemHealth]:
        start = time.time()
        try:
            # Import lazily to avoid mandatory dependency
            from ai.allocators.portfolio_brain import PortfolioBrain  # type: ignore

            # We just check that the class is importable; we do not construct full instance
            latency_ms = (time.time() - start) * 1000
            return SubsystemHealth(
                name="portfolio_brain",
                status="ok",
                latency_ms=latency_ms,
                meta={"cls": getattr(PortfolioBrain, "__name__", "PortfolioBrain")},
            )
        except Exception as e:
            return SubsystemHealth(name="portfolio_brain", status="fail", error=str(e))
