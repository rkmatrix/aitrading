# Bot Tuning Changes - January 30, 2026

## Summary
Implemented critical threshold adjustments to enable the bot to execute trades. The bot was functioning correctly but thresholds were too conservative for current market conditions.

## Changes Made

### 1. Entry Threshold Lowered ✅
**File:** `runner/phase26_realtime_live.py`
- **Before:** Base entry threshold = 0.08 (8%)
- **After:** Base entry threshold = 0.03 (3%)
- **Impact:** Signals with 3-4% strength can now pass the entry threshold
- **Line:** 1762

**Why:** 
- Actual signal strengths are 3.0-4.0% (AAPL: 3.30%, MSFT: 3.91%, TSLA: 3.29%)
- Previous threshold of 12% was 3-4x higher than actual signals
- This was the primary blocker preventing all trades

### 2. Spread Filter Relaxed ✅
**File:** `configs/trade_quality.yaml`
- **Before:** `max_bid_ask_spread_bps: 10.0` (0.1% spread)
- **After:** `max_bid_ask_spread_bps: 20.0` (0.2% spread)
- **Impact:** More signals will pass the spread filter during normal market conditions
- **Line:** 42

**Why:**
- Many valid signals were rejected due to spreads of 15-350+ bps
- 10 bps is too strict for normal market conditions
- 20 bps allows reasonable spreads while still filtering extreme cases

## Expected Behavior After Changes

### Before Changes:
- ✅ Signals generated: 3.0-4.0%
- ✅ Trade Quality Filter: Passing (threshold 0.03)
- ❌ Entry Threshold: Blocking (threshold 0.12)
- ❌ Spread Filter: Blocking many signals (threshold 10 bps)
- **Result:** 0 trades in 18,700+ ticks

### After Changes:
- ✅ Signals generated: 3.0-4.0%
- ✅ Trade Quality Filter: Passing (threshold 0.03)
- ✅ Entry Threshold: Should pass (threshold ~0.03-0.04)
- ✅ Spread Filter: More signals passing (threshold 20 bps)
- **Expected Result:** Bot should start executing trades

## Next Steps

1. **Restart the bot** to apply changes:
   ```powershell
   # Stop current bot (Ctrl+C)
   # Then restart:
   python runner\phase26_realtime_live.py
   # Or use batch file:
   start_bot.bat
   ```

2. **Monitor for 1-2 hours** to observe:
   - Diagnostic summaries showing trades passing filters
   - Actual trade executions
   - Trade frequency (expect 1-5 trades per day conservatively)

3. **Review diagnostic summaries** (every 100 ticks):
   - Check "Ticks below entry threshold" - should decrease significantly
   - Check "Ticks filtered by Trade Quality" - spread rejections should decrease
   - Monitor account equity changes

4. **Fine-tune if needed**:
   - If too many trades: Increase `PHASE26_BASE_ENTRY_THR` to 0.04-0.05
   - If still no trades: Check diagnostic summaries for other blockers
   - If spreads still blocking: Consider increasing to 25 bps

## Technical Details

### Entry Threshold Calculation
The entry threshold is dynamically calculated:
```
entry_threshold = base_thr * (1.0 + vol * 0.5 + dd * 2.0)
```

With base_thr = 0.03:
- Low vol/dd: ~0.03-0.04 (3-4%)
- Medium vol/dd: ~0.05-0.08 (5-8%)
- High vol/dd: ~0.10-0.15 (10-15%)
- Clamped between 0.02 and 0.25

### Spread Filter
- Checks bid-ask spread in basis points (bps)
- 1 bps = 0.01% = $0.01 per $100 stock price
- 20 bps = 0.2% = $0.20 per $100 stock price
- Filters out illiquid stocks with excessive spreads

## Rollback Instructions

If you need to revert these changes:

1. **Entry Threshold:**
   ```python
   # In runner/phase26_realtime_live.py line 1762:
   base_thr = float(os.getenv("PHASE26_BASE_ENTRY_THR", "0.08"))
   ```

2. **Spread Filter:**
   ```yaml
   # In configs/trade_quality.yaml line 42:
   max_bid_ask_spread_bps: 10.0
   ```

## Notes

- These changes make the bot more aggressive (more trades)
- Monitor performance closely for the first few days
- The bot still has multiple safety layers:
  - Trade Quality Filter (min signal strength, win rate, etc.)
  - Risk management (max equity risk, cooldowns)
  - Dynamic threshold adjustment (increases with volatility/drawdown)
  - Portfolio-level risk controls

- The bot was working correctly before - thresholds were just too conservative
- These changes align thresholds with actual market signal strengths
