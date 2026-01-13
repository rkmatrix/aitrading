# ai/evolution/engine.py
from __future__ import annotations
import json, shutil, uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ai.policy.pool import PolicyBundle, PolicyPool
from ai.evolution.mutators import mutate_hparams, crossover_hparams
from ai.evolution.archive import EvolutionArchive

class EvolutionEngine:
    def __init__(self, bundles_dir: Path, pool: PolicyPool, archive: EvolutionArchive, spec: Dict[str, Any], mutate_prob: float, crossover_prob: float, max_pool_size: int, elite_top_k: int, sample_new_k: int):
        self.bundles_dir = Path(bundles_dir)
        self.pool = pool
        self.archive = archive
        self.spec = spec
        self.mutate_prob = mutate_prob
        self.crossover_prob = crossover_prob
        self.max_pool_size = max_pool_size
        self.elite_top_k = elite_top_k
        self.sample_new_k = sample_new_k

    def _write_bundle(self, manifest: Dict[str, Any]) -> PolicyBundle:
        pid = manifest["policy_id"]
        out_dir = self.bundles_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        # (optional) weights can be produced by training jobs later
        b = PolicyBundle.load_from_dir(out_dir)
        return b

    def evolve(self, ranked: List[Tuple[str, float]]) -> List[PolicyBundle]:
        """
        ranked: list of (policy_id, score) best first
        """
        new_bundles: List[PolicyBundle] = []
        elites = ranked[: max(1, self.elite_top_k)]
        elite_ids = [pid for pid, _ in elites]

        # Mutations of elites
        for pid, score in elites:
            parent = self.pool.get(pid)
            if not parent: 
                continue
            m = json.loads(json.dumps(parent.manifest))
            hp = m.get("hyperparams", {})
            m["hyperparams"] = mutate_hparams(hp, self.spec, self.mutate_prob)
            m["policy_id"] = f"{pid}_mut_{uuid.uuid4().hex[:6]}"
            m["lineage"] = {"parent": pid, "type": "mutation"}
            nb = self._write_bundle(m)
            self.pool.add(nb)
            new_bundles.append(nb)
            self.archive.log_event("mutate", nb.policy_id, parent=pid, notes="mutation of elite")

        # Crossovers between top 2 elites (if possible)
        if len(elites) >= 2:
            a_id, _ = elites[0]
            b_id, _ = elites[1]
            a = self.pool.get(a_id); b = self.pool.get(b_id)
            if a and b:
                am = json.loads(json.dumps(a.manifest))
                bm = json.loads(json.dumps(b.manifest))
                hpc = crossover_hparams(am.get("hyperparams", {}), bm.get("hyperparams", {}), self.spec, self.crossover_prob)
                nm = json.loads(json.dumps(am))
                nm["hyperparams"] = hpc
                nm["policy_id"] = f"{a.policy_id}x{b.policy_id}_cx_{uuid.uuid4().hex[:6]}"
                nm["lineage"] = {"parents": [a.policy_id, b.policy_id], "type": "crossover"}
                nb = self._write_bundle(nm)
                self.pool.add(nb)
                new_bundles.append(nb)
                self.archive.log_event("crossover", nb.policy_id, parent=f"{a.policy_id}|{b.policy_id}", notes="crossover of elites")

        # Random samples (new seeds)
        for _ in range(max(0, self.sample_new_k)):
            base = self.pool.get(elite_ids[0]) if elite_ids else None
            if not base: 
                break
            nm = json.loads(json.dumps(base.manifest))
            nm["policy_id"] = f"{base.policy_id}_seed_{uuid.uuid4().hex[:6]}"
            # randomize listed hparams inside bounds
            hp = nm.get("hyperparams", {})
            for k, s in self.spec.items():
                if k in hp:
                    low, high = float(s.get("low", hp[k])), float(s.get("high", hp[k]))
                    if s.get("log", False):
                        import math, random
                        lv_low, lv_high = math.log(low), math.log(high)
                        hp[k] = float(math.exp(random.uniform(lv_low, lv_high)))
                    else:
                        import random
                        hp[k] = float(random.uniform(low, high))
            nm["hyperparams"] = hp
            nm["lineage"] = {"parent": base.policy_id, "type": "random_seed"}
            nb = self._write_bundle(nm)
            self.pool.add(nb)
            new_bundles.append(nb)
            self.archive.log_event("seed", nb.policy_id, parent=base.policy_id, notes="randomized new seed")

        # Pool trimming handled by pool.max_size; also ensure bundles_dir stays tidy (optional)
        return new_bundles
