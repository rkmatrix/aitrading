"""
ai/train/train_slippage_predictor.py
Phase 73 — Train SlippagePredictor from trade_journal.csv
"""

import csv
import logging
from pathlib import Path

from ai.models.slippage_predictor import SlippagePredictor

logger = logging.getLogger(__name__)
JOURNAL_PATH = Path("data/reports/trade_journal.csv")


def load_journal_rows():
    rows = []
    if not JOURNAL_PATH.exists():
        raise FileNotFoundError(f"trade_journal.csv not found at {JOURNAL_PATH}")

    with JOURNAL_PATH.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def train_slippage():
    rows = load_journal_rows()
    logger.info("Phase 73: Loaded %d journal rows", len(rows))

    model = SlippagePredictor()
    model.train(rows)
    model.save()

    logger.info("Phase 73: SlippagePredictor saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_slippage()
