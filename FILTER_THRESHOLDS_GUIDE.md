# Trading Bot Filter Thresholds Guide

This document explains all the filters and thresholds that control when the bot executes trades.

## Overview

The bot uses **multiple layers of filters** to ensure only high-quality trades are executed. This conservative approach protects capital but may result in fewer trades. Understanding these thresholds helps you tune the bot's behavior.

## Filter Layers (In Order)

1. **Price Fetching** - Must successfully get market prices
2. **Trade Quality Filter** - Statistical edge validation
3. **Entry Threshold** - Dynamic signal strength requirement
4. **MultiSymbolAllocator** - Portfolio-aware allocation
5. **Risk Envelope** - Final risk checks before execution

---

## 1. Price Fetching

**Location**: `runner/phase26_realtime_live.py` → `_get_last_price()`

**What it does**: Fetches the latest price from Alpaca API

**Why trades might be blocked**:
- Market is closed
- API rate limits exceeded
- Network issues
- Symbol not available

**Diagnostics**: Check logs for "Price fetch failed" or "No prices available"

**Fix**: Ensure market is open, check API connectivity, verify symbols are valid

---

## 2. Trade Quality Filter

**Location**: `configs/trade_quality.yaml`

**Purpose**: Only trade when there's a statistical edge

### Key Thresholds:

#### `min_signal_strength: 0.05` (5%)
- **Meaning**: Minimum absolute signal strength required
- **Range**: 0.01 - 0.20
- **Default**: 0.05 (conservative)
- **Adjustment**:
  - Lower to 0.02-0.03 for more trades
  - Raise to 0.08-0.10 for stronger signals only

#### `min_win_rate: 0.45` (45%)
- **Meaning**: Minimum historical win rate required
- **Range**: 0.0 - 1.0 (0% - 100%)
- **Default**: 0.45 (45% win rate)
- **Adjustment**:
  - Lower to 0.35-0.40 for more opportunities
  - Raise to 0.50+ for higher quality trades

#### `min_expected_value_pct: 0.5` (0.5%)
- **Meaning**: Minimum expected value (%) a trade must have
- **Range**: 0.0 - 10.0
- **Default**: 0.5% (conservative)
- **Adjustment**: Lower to 0.2-0.3 for more aggressive trading

#### `min_volume: 1000000` ($1M daily volume)
- **Meaning**: Minimum daily trading volume to ensure liquidity
- **Range**: 100,000 - 10,000,000
- **Default**: 1,000,000 (avoids illiquid stocks)
- **Adjustment**: Lower to 500k for more opportunities

#### `max_bid_ask_spread_bps: 10.0` (0.1%)
- **Meaning**: Maximum bid-ask spread in basis points
- **Range**: 1.0 - 50.0
- **Default**: 10.0 bps (0.1% spread)
- **Adjustment**: Lower to 5.0 for tighter spreads

---

## 3. Entry Threshold (Dynamic)

**Location**: `runner/phase26_realtime_live.py` → `_compute_stability_params()`

**Purpose**: Dynamic threshold that adjusts based on market conditions

**How it works**:
- **Base**: 0.08 (8%) - configurable via `PHASE26_BASE_ENTRY_THR` env var
- **Increases with volatility**: Higher vol → Higher threshold (more selective)
- **Increases with drawdown**: Higher DD → Higher threshold (more defensive)
- **Range**: 0.02 (2%) to 0.25 (25%) - clamped for safety

**Formula**: `entry_threshold = base_thr * (1.0 + vol * 0.5 + dd * 2.0)`

**Adjustment**:
- Set `PHASE26_BASE_ENTRY_THR=0.05` for more trades (lower threshold)
- Set `PHASE26_BASE_ENTRY_THR=0.12` for fewer trades (higher threshold)
- Lower values = more aggressive, higher values = more conservative

**Why trades might be blocked**: Signal strength below the dynamic threshold

---

## 4. MultiSymbolAllocator Filter

**Location**: `ai/allocation/multi_symbol_allocator.py`

**Purpose**: Portfolio-aware allocation that selects best symbols

### Key Thresholds:

#### `min_abs_score: 0.05` (5%)
- **Meaning**: Minimum absolute score to consider for allocation
- **Configurable**: Via `PHASE28_MIN_ABS_SCORE` env var
- **Default**: 0.05
- **Adjustment**: Lower to 0.02-0.03 for more symbols, raise to 0.08+ for stronger signals

#### `max_active_symbols: 3`
- **Meaning**: Maximum number of symbols to trade simultaneously
- **Configurable**: Via `PHASE28_MAX_ACTIVE` env var
- **Default**: 3
- **Adjustment**: Increase for more diversification, decrease for focus

---

## 5. Signal Floor (Micro Allocator)

**Location**: `configs/phase27_micro_alloc.yaml`

**Purpose**: Filters out weak signals before allocation

#### `signal_floor: 0.05` (5%)
- **Meaning**: Minimum absolute fused score to consider
- **Range**: 0.01 - 0.20
- **Default**: 0.05
- **MEANING**: Signals with `abs(fused_score) < signal_floor` are ignored
- **This is the PRIMARY filter preventing weak signals from becoming trades**

**Adjustment**:
- Lower to 0.02-0.03 for more trades
- Raise to 0.08-0.10 for stronger signals only

---

## Diagnostic Summary

The bot logs a diagnostic summary every 100 ticks showing:
- How many ticks were blocked by each filter
- Price fetch failures by symbol
- Quality filter rejection reasons
- Current threshold values
- Account status

**To see diagnostics**: Check logs for "DIAGNOSTIC SUMMARY" entries

---

## Quick Tuning Guide

### If bot is too conservative (not trading enough):

1. **Lower Trade Quality thresholds**:
   ```yaml
   min_signal_strength: 0.03  # from 0.05
   min_win_rate: 0.40  # from 0.45
   ```

2. **Lower Entry Threshold base**:
   ```bash
   export PHASE26_BASE_ENTRY_THR=0.05  # from 0.08
   ```

3. **Lower Signal Floor**:
   ```yaml
   signal_floor: 0.03  # from 0.05
   ```

4. **Lower MultiSymbolAllocator threshold**:
   ```bash
   export PHASE28_MIN_ABS_SCORE=0.03  # from 0.05
   ```

### If bot is too aggressive (trading too much):

1. **Raise Trade Quality thresholds**:
   ```yaml
   min_signal_strength: 0.08  # from 0.05
   min_win_rate: 0.50  # from 0.45
   ```

2. **Raise Entry Threshold base**:
   ```bash
   export PHASE26_BASE_ENTRY_THR=0.12  # from 0.08
   ```

3. **Raise Signal Floor**:
   ```yaml
   signal_floor: 0.08  # from 0.05
   ```

---

## Understanding Why Trades Are Blocked

Check the diagnostic summary in logs (every 100 ticks) to see:
1. **Price failures**: API issues or market closed
2. **Quality filter rejections**: Which specific reason (low signal strength, low win rate, etc.)
3. **Threshold rejections**: Signal scores vs entry threshold
4. **Allocator rejections**: Portfolio allocation decisions

---

## Important Notes

- **Conservative behavior is GOOD**: The bot is protecting your capital
- **Multiple filters work together**: A signal must pass ALL filters to execute
- **Dynamic thresholds**: Entry threshold adjusts automatically based on market conditions
- **Start conservative**: It's easier to lower thresholds than recover from bad trades
- **Monitor diagnostics**: Use the diagnostic summary to understand bot behavior

---

## Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `PHASE26_BASE_ENTRY_THR` | 0.08 | Base entry threshold (8%) |
| `PHASE26_MAX_EQUITY_RISK_PCT` | 0.03 | Max equity risk per trade (3%) |
| `PHASE26_SYMBOL_COOLDOWN_SEC` | 60.0 | Cooldown between trades (seconds) |
| `PHASE28_MIN_ABS_SCORE` | 0.05 | MultiSymbolAllocator min score (5%) |
| `PHASE28_MAX_ACTIVE` | 3 | Max active symbols |

---

## Questions?

If the bot isn't trading:
1. Check diagnostic summary in logs
2. Verify account has buying power (not $0 equity)
3. Ensure market is open
4. Review filter thresholds vs actual signal scores
5. Check for API connectivity issues
