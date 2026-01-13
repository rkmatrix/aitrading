from ..simulators.market_sim import MicroMarketSim
from ..policies.rule_based import TWAP, POV
from ..policies.bandit import LinUCB
from ..cost_models.almgren_chriss import AlmgrenChrissCost

def run_benchmarks(symbol="AAPL", total_qty=5000):
    sim = MicroMarketSim()
    cost = AlmgrenChrissCost(0.0005, 0.15, 0.05)
    # Simple bench loop comparing policies:
    benches = {
        "twap": TWAP(total_qty, steps=50),
        "pov": POV(0.1, min_size=100),
        "bandit": LinUCB()
    }
    results = {}
    for name, pol in benches.items():
        sim = MicroMarketSim()
        filled = 0; fills=[]
        for _ in range(50):
            s = sim.snapshot()
            st = {"spread": s["spread"], "imbalance": s["imbalance"], "volatility": s["volatility"],
                  "remaining_time": 1.0, "remaining_qty": total_qty - filled, "micro_alpha": 0.0,
                  "est_volume": 1000}
            a = pol.act(st)
            qty = min(a["size"], total_qty - filled)
            px, got, s2 = sim.execute(qty, ["passive","mid","market"].index(a["aggression"]))
            fills.append((got, px))
            filled += got
            if filled >= total_qty: break
        results[name] = {"filled": filled, "fills": fills}
    return results
