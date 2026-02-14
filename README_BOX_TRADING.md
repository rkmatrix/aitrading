# Box Trading Bot - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Strategy Explanation](#strategy-explanation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Understanding the Alerts](#understanding-the-alerts)
6. [Phased Rollout Plan](#phased-rollout-plan)
7. [Performance Monitoring](#performance-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Overview

The **Box Trading Bot** implements a mean-reversion strategy based on the previous day's high and low prices. The core concept is simple but effective when applied in the right market conditions with proper risk management.

### What It Does
- Identifies support (previous day's low) and resistance (previous day's high) levels
- Buys when price reaches the bottom zone (near previous day's low)
- Sells when price reaches the top zone (near previous day's high)
- Exits at predefined targets or stop losses
- Only trades during regular market hours (9:30 AM - 4:00 PM ET)
- Sends detailed Telegram alerts for every trade

### What Makes It Different
This implementation includes **12 layers of protection** not found in basic box trading systems:
- ✅ Breakout detection (avoids fighting trends)
- ✅ Whipsaw protection (prevents revenge trading)
- ✅ Multi-layer regime filtering (only trades when conditions are right)
- ✅ Adaptive zone thresholds (adjusts to volatility)
- ✅ Correlation management (prevents cascade failures)
- ✅ Time-based exits (doesn't hold losers)
- ✅ Multiple circuit breakers (hard limits on losses)
- ✅ Performance-based learning (improves over time)

---

## Strategy Explanation

### The "Box" Concept

Every trading day, we draw a "box" using:
- **Top of box**: Previous day's high price (strongest sellers)
- **Bottom of box**: Previous day's low price (strongest buyers)
- **Midpoint**: Center of the range (neutral zone)

```
Previous Day High ─────────┐  ← Top Zone (SELL HERE)
                          │
                          │  Middle Zone (AVOID)
Midpoint ..................│
                          │  Middle Zone (AVOID)
                          │
Previous Day Low ──────────┘  ← Bottom Zone (BUY HERE)
```

### Trading Rules

#### Rule #1: Don't Buy at the Top
When price is near the previous day's high, **don't buy**. This is where the strongest sellers showed up yesterday, and they'll likely return.

#### Rule #2: Don't Sell at the Bottom
When price is near the previous day's low, **don't sell**. This is where the strongest buyers showed up yesterday, and they'll likely return.

#### Rule #3: Don't Trade in the Middle
The middle 40% of the range is a no-trade zone. Here, the probability is 50/50 - no edge. We only trade at the boundaries where we have statistical advantage.

### Entry Signals

**BUY Signal** requires ALL of:
1. Price within bottom zone (default: within 0.5% of previous day's low)
2. RSI oversold (< 30)
3. Volume spike (>50% above average)
4. Rejection candle (wick showing rejection from lows)
5. No strong downward momentum
6. Market in rangebound regime
7. Symbol not blacklisted

**SELL Signal** requires ALL of:
1. Price within top zone (default: within 0.5% of previous day's high)
2. RSI overbought (> 70)
3. Volume spike (>50% above average)
4. Rejection candle (wick showing rejection from highs)
5. No strong upward momentum
6. Market in rangebound regime
7. Symbol not blacklisted

### Exit Strategy

**Tiered Profit Taking:**
1. Exit 50% of position at midpoint (lock in profits)
2. Exit 30% at 75% of range
3. Exit final 20% at opposite box boundary

**Stop Loss:**
- Placed just beyond the box boundary
- Example: For long, stop below previous day's low

**Time-Based Exit:**
- Force close after 2 hours if not profitable
- Mean reversion should happen quickly - if it doesn't, exit

### Why It Works (When It Does)

**Market Psychology:**
- Previous day's high/low represent significant price discovery
- Large institutional orders often cluster at these levels
- Support/resistance levels are self-fulfilling (traders watch them)

**Statistical Edge:**
- In ranging markets, price has ~70-80% chance of reverting from extremes
- Risk/reward is favorable (small risk, large reward)
- Probability is on your side at the boundaries

**When It Fails:**
- **Trending markets**: Price breaks through and doesn't come back
- **Breakouts**: Volume surge + momentum = new trend starting
- **Low volume**: No institutional participation = weak levels
- **Gaps**: Overnight news changes the game

**Our Solution:**
We detect these failure modes and **skip trading** when they occur.

---

## Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Alpaca account** (paper trading or live)
3. **Telegram bot** setup (for alerts)
4. **.env file** configured with API keys

### Installation

No additional installation needed - uses existing bot infrastructure.

### Configuration

1. **Check your .env file:**
```env
ENV=PAPER_TRADING  # Start with paper trading!
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

2. **Review box_trading.yaml:**
```bash
# Open and review the config
notepad configs\box_trading.yaml
```

Key settings to check:
- `symbols`: List of stocks to trade
- `max_positions`: Start with 2
- `base_risk_per_trade`: 0.02 (2% of capital)
- `current_phase`: "paper_testing"

3. **Start the bot:**
```bash
# Double-click the batch file
boxTradingbot.bat

# Or run directly
python runner\box_trading_runner.py
```

### What Happens Next

1. Bot starts and sends Telegram alert
2. Checks if market is open
3. If open, scans symbols for signals
4. Executes trades when conditions met
5. Manages positions until exit
6. Sends alerts for every action
7. Sends daily summary at 4:05 PM ET
8. Idles when market closed

---

## Configuration

### Key Settings Explained

#### Symbol Selection
```yaml
symbols:
  - SPY    # Most liquid, best for testing
  - QQQ    # Tech exposure
  - AAPL   # Large cap
  - MSFT   # Large cap
```

**Recommendations:**
- Start with 2-4 symbols
- Use highly liquid symbols (>10M daily volume)
- Avoid penny stocks and low-volume names
- Mix indices and individual stocks

#### Zone Thresholds

**Adaptive Mode** (recommended):
```yaml
zone_calculation_mode: "adaptive"
adaptive_zones:
  low_volatility:   # ATR < 1%
    top_threshold: 0.003    # 0.3% zone
  medium_volatility:  # ATR 1-2%
    top_threshold: 0.007    # 0.7% zone
  high_volatility:  # ATR > 2%
    top_threshold: 0.012    # 1.2% zone
```

Zones automatically widen in high volatility, tighten in low volatility.

**Fixed Mode** (simpler):
```yaml
zone_calculation_mode: "fixed"
fixed_top_zone_threshold: 0.005  # Always 0.5%
fixed_bottom_zone_threshold: 0.005
```

#### Risk Management

```yaml
base_risk_per_trade: 0.02    # 2% of capital per trade
max_risk_per_trade: 0.04     # 4% maximum (high confidence)
min_risk_per_trade: 0.01     # 1% minimum (low confidence)

max_positions: 2             # Maximum concurrent positions
```

**How Position Sizing Works:**
1. Account equity: $10,000
2. Risk per trade: 2% = $200
3. Stop distance: $1.00
4. Position size: $200 / $1.00 = 200 shares
5. Entry price: $50, Stop: $49
6. Max loss if stopped: $200 (2%)

#### Circuit Breakers

```yaml
circuit_breakers:
  max_daily_loss_percent: 0.05       # Stop at 5% daily loss
  max_consecutive_losses: 3          # Pause after 3 losses
  max_daily_trades: 15              # Prevent overtrading
  max_drawdown_from_peak: 0.10      # Stop at 10% drawdown
```

These are **hard limits** that cannot be overridden by the bot.

---

## Understanding the Alerts

### Trade Entry Alert
```
🎯 BOX TRADE ENTERED

Symbol: AAPL
Action: BUY
Entry: $175.20
Quantity: 50 shares
Stop Loss: $174.50
Targets: $176.80 / $177.50 / $178.30
Risk: $35.00
R:R Ratio: 2.29:1

Box Levels:
- Prev High: $178.40
- Prev Low: $174.60
- Midpoint: $176.50
- Range: 2.17%

Confidence: 85%
Reasons: Price in bottom zone, RSI oversold (28.5), 
         Volume confirmation (1.8x avg), Second touch of bottom
Time: 2026-02-11 10:45:32 ET
```

**What This Tells You:**
- Symbol and direction
- Exact entry price and quantity
- Where stop loss is placed
- Three profit targets
- Dollar risk on this trade
- Risk/reward ratio
- Box boundaries for context
- Why the trade was taken
- Confidence level (75-100%)

### Trade Exit Alert
```
✅ BOX TRADE CLOSED

Symbol: AAPL
Action: BUY
Entry: $175.20
Exit: $176.80
Quantity: 50 shares
P&L: $80.00 (+1.83%)
Duration: 45 minutes
Reason: Target 1 (Midpoint)
Time: 2026-02-11 11:30:15 ET
```

**What This Tells You:**
- Final P&L in dollars and percent
- How long trade lasted
- Why it was closed (target hit, stop loss, time limit, etc.)

### Circuit Breaker Alert
```
🛑 DAILY LOSS LIMIT HIT

Loss: $-512.00 (5.12%)
Limit: 5%

Trading stopped for today.
```

**What This Tells You:**
- Which limit was hit
- Current loss amount
- What action was taken

### Daily Summary Alert
```
📊 BOX TRADING - DAILY SUMMARY

Date: 2026-02-11

Trades: 8
Wins: 5
Losses: 3
Win Rate: 62.5%

Total P&L: $245.00
Max Drawdown: $85.00

Overall Stats (All Time):
Total Trades: 127
Win Rate: 58.3%
Total P&L: $3,420.00
Avg Win: $95.50
Avg Loss: $48.25
```

**What This Tells You:**
- Daily performance
- All-time cumulative stats
- Win/loss statistics
- Average trade size

---

## Phased Rollout Plan

### Phase 1: Paper Trading (Weeks 1-2)
**Goal:** Validate strategy and find bugs

**Settings:**
```yaml
current_phase: "paper_testing"
symbols: [SPY]  # Single symbol
max_positions: 1
```

**Success Criteria:**
- ✅ Bot runs without crashes
- ✅ Telegram alerts working
- ✅ Trades executed correctly
- ✅ 20+ trades completed
- ✅ No critical bugs

**Action:** Monitor every trade, document issues

---

### Phase 2: Extended Paper Trading (Weeks 3-4)
**Goal:** Prove statistical edge

**Settings:**
```yaml
current_phase: "paper_testing"
symbols: [SPY, QQQ, AAPL, MSFT]  # Expand symbols
max_positions: 2
```

**Success Criteria:**
- ✅ 50+ trades total
- ✅ Win rate >55%
- ✅ Profit factor >1.4
- ✅ Max drawdown <8%
- ✅ All circuit breakers tested
- ✅ Strategy understood deeply

**Action:** Analyze performance by symbol, hour, condition

---

### Phase 3: Live Phase 1 (Weeks 5-6)
**Goal:** Prove execution in live environment

**Settings:**
```yaml
current_phase: "live_phase1"
symbols: [SPY]  # Back to single symbol
max_positions: 1
phase_settings:
  live_phase1:
    max_position_size_dollars: 100  # TINY positions
```

⚠️ **IMPORTANT:** Change ENV to LIVE in .env file

**Success Criteria:**
- ✅ 10+ live trades
- ✅ Win rate >50%
- ✅ No execution errors
- ✅ Slippage acceptable
- ✅ Commissions factored in
- ✅ Emotional control maintained

**Action:** Monitor intensely, document real-world differences

---

### Phase 4: Live Phase 2 (Weeks 7-10)
**Goal:** Scale to production size

**Settings:**
```yaml
current_phase: "live_phase2"
symbols: [SPY, QQQ, AAPL, MSFT]
max_positions: 2
phase_settings:
  live_phase2:
    max_position_size_dollars: 500  # 50% of intended size
```

**Success Criteria:**
- ✅ 30+ live trades
- ✅ Win rate >55%
- ✅ Consistent profitability
- ✅ No major issues
- ✅ Comfortable with system

**Action:** Continue monitoring, fine-tune parameters

---

### Phase 5: Full Production (Week 11+)
**Goal:** Full deployment

**Settings:**
```yaml
current_phase: "production"
symbols: [SPY, QQQ, AAPL, MSFT]  # Or expand more
max_positions: 2  # Can increase to 3 if win rate >65%
phase_settings:
  production:
    max_position_size_dollars: 5000  # Full size
```

**Success Criteria:**
- ✅ Proven track record
- ✅ Consistent profits
- ✅ Win rate maintained
- ✅ Drawdowns controlled

**Action:** Monthly reviews, continuous improvement

---

## Performance Monitoring

### Daily Checks

**Every Morning:**
1. Check Telegram for daily summary from previous day
2. Review any circuit breaker alerts
3. Check log file for errors: `data/logs/box_trading_bot.log`
4. Verify bot is running (if market open)

**Every Evening:**
1. Review daily summary alert
2. Analyze losing trades (what went wrong?)
3. Check if any symbols should be blacklisted
4. Plan any configuration changes

### Weekly Analysis

**Check These Metrics:**
1. **Win Rate**: Should be >55%
2. **Profit Factor**: Avg win / Avg loss, should be >1.4
3. **Max Drawdown**: Should be <10%
4. **Best/Worst Symbols**: Which symbols working?
5. **Best/Worst Hours**: Which times working?
6. **Average Hold Time**: Should be <2 hours
7. **Stop Out Rate**: How often are we stopped?

**Action Items:**
- Remove consistently losing symbols
- Blacklist consistently losing hours
- Adjust zone thresholds if needed
- Tighten confirmations if too many false signals

### Monthly Review

**Deep Dive Analysis:**
1. Export all trade data
2. Calculate:
   - Sharpe ratio
   - Maximum consecutive losses
   - Largest loss vs average loss
   - Win rate by symbol, hour, day of week
3. Compare vs benchmark (buy and hold SPY)
4. Identify patterns in losses
5. Test parameter adjustments

**Questions to Ask:**
- Is the strategy still working?
- Are market conditions changing?
- Do we need to adjust zone sizes?
- Should we add/remove symbols?
- Are confirmations too strict/loose?

---

## Troubleshooting

### Bot Won't Start

**Error: "Python is not installed"**
- Install Python 3.8+ from python.org
- Add Python to PATH during installation
- Restart command prompt

**Error: ".env file not found"**
- Create .env file in project root
- Copy from .env.example if available
- Add API keys and configuration

**Error: "Module not found"**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Error: "Configuration file not found"**
- Ensure configs/box_trading.yaml exists
- Check file name spelling
- Verify you're in correct directory

### No Trades Being Executed

**Possible Reasons:**

1. **Market Closed**
   - Bot only trades 9:30 AM - 3:45 PM ET
   - Will idle when market closed
   - Check market calendar for holidays

2. **No Valid Signals**
   - All confirmations must pass
   - Price must be in top/bottom zone
   - Symbol must not be blacklisted
   - Check log file for "skipped" messages

3. **Circuit Breaker Active**
   - Check if daily loss limit hit
   - Check if paused after consecutive losses
   - Check Telegram for circuit breaker alerts
   - Wait for reset (usually next day)

4. **Trending Market**
   - Strategy works in ranging markets
   - Breakout detection prevents trading during trends
   - This is WORKING AS INTENDED
   - Be patient for ranging conditions

### Trades Losing Money

**First: Don't Panic**
- Some losses are normal
- Strategy targets 55-65% win rate
- Judge performance over 20+ trades, not 2-3

**Check These:**
1. **Win Rate**: If <45%, pause and analyze
2. **Market Conditions**: Strong trend? High volatility?
3. **Symbol Selection**: Are some symbols consistently losing?
4. **Zone Sizes**: Too tight? Too loose?
5. **Hold Times**: Holding too long?

**Action Steps:**
1. Review last 10 trades in detail
2. Categorize losses (stopped out, time-based, etc.)
3. Look for patterns
4. Adjust configuration or pause bot
5. Return to paper trading if needed

### Telegram Alerts Not Working

**Check:**
1. TELEGRAM_BOT_TOKEN in .env
2. TELEGRAM_CHAT_ID in .env
3. Telegram settings in box_trading.yaml:
   ```yaml
   telegram:
     enabled: true
   ```
4. Test manually:
   ```python
   from tools.telegram_alerts import notify
   notify("Test message", kind="orders")
   ```

### Position Stuck Open

**Manual Intervention:**
1. Log into Alpaca dashboard
2. Close position manually
3. Restart bot
4. Check why auto-exit failed (review logs)

---

## FAQ

### Q: How much money do I need to start?
**A:** 
- Paper trading: $0 (simulated)
- Live trading: Minimum $2,000 recommended
- With $500 risk per trade (2% risk), need $25,000
- Start smaller: $100-500 position sizes in Phase 1

### Q: What returns should I expect?
**A:** 
- **Realistic**: 1-3% per month
- **Good month**: 5-8%
- **Bad month**: -2 to -5%
- **Not realistic**: "$1,000/day" or "70-80% win rate"

### Q: How many trades per day?
**A:**
- Typical: 2-5 trades
- Slow day: 0-1 trades
- Active day: 5-10 trades
- Circuit breaker limit: 15 trades max

### Q: Do I need to monitor it constantly?
**A:**
- Paper trading: Check daily
- Live Phase 1: Check every trade
- Live Phase 2: Check 2-3 times per day
- Production: Daily review sufficient
- Telegram alerts keep you informed

### Q: What if I want to stop the bot?
**A:**
1. Press Ctrl+C in the command window
2. Bot will close all positions
3. Send final summary via Telegram
4. Safe shutdown

### Q: Can I run this alongside my existing bot?
**A:**
- Yes, they're completely separate
- Use different symbols to avoid conflicts
- Example: Main bot trades TSLA, NVDA; Box bot trades SPY, QQQ
- Both use same broker account (OK)

### Q: What about overnight positions?
**A:**
- Bot NEVER holds overnight
- All positions closed by 3:55 PM ET
- This is by design (avoid gap risk)
- No exceptions

### Q: Why did it skip a signal?
**A:**
Common reasons in logs:
- "In middle zone" - Price not at boundary
- "RSI not oversold" - Missing confirmation
- "Low volume" - Institutional participation lacking
- "Too much momentum" - Might be breakout
- "Blacklisted" - Recent stop-out
- "Breakout detected" - Trending, not ranging

### Q: How do I add more symbols?
**A:**
1. Edit configs/box_trading.yaml
2. Add to symbols list
3. Ensure symbol is liquid (>10M daily volume)
4. Add to appropriate correlation_group
5. Restart bot
6. Monitor new symbol performance

### Q: Can I paper trade and live trade simultaneously?
**A:**
- Not recommended (same code, same config)
- Run separate instances with different configs
- Or use different accounts
- Better: Graduate from paper to live sequentially

### Q: What's the best time of day to trade?
**A:**
- Usually: 10:00 AM - 2:00 PM ET
- Avoid: First 30 minutes (volatile)
- Avoid: Last 30 minutes (unpredictable)
- Bot tracks performance by hour
- After 20+ trades, review "best_hours" in stats

### Q: Help! It's down $500 today!
**A:**
1. Check if circuit breaker triggered (should have)
2. If not, check max_daily_loss_percent setting
3. Review trades - trending market?
4. If settings correct, this shouldn't happen
5. Contact for support if circuit breaker failed

---

## Support

### Getting Help

1. **Check the logs first:**
   ```
   data/logs/box_trading_bot.log
   ```

2. **Review this README**

3. **Check Telegram alerts** for clues

4. **GitHub Issues** (if open source)

5. **Community Discord** (if available)

### Reporting Bugs

Include:
1. Full error message
2. Relevant log excerpts
3. Configuration (sensitive data removed)
4. Steps to reproduce
5. Expected vs actual behavior

---

## Disclaimer

⚠️ **IMPORTANT DISCLAIMERS**

1. **No Guarantee of Profit**
   - Past performance ≠ future results
   - You can lose money trading
   - Never trade money you can't afford to lose

2. **Your Responsibility**
   - You are responsible for monitoring the bot
   - You are responsible for all trades executed
   - Review and understand the code before use
   - Test thoroughly in paper trading

3. **Not Financial Advice**
   - This is educational software
   - Not personalized investment advice
   - Consult a financial advisor
   - Understand the risks

4. **Market Risk**
   - Markets are unpredictable
   - Black swan events happen
   - Circuit breakers help but aren't perfect
   - Always use stop losses

5. **Technical Risk**
   - Software can have bugs
   - Internet can disconnect
   - APIs can fail
   - Have backup plans

**By using this software, you acknowledge these risks and take full responsibility for any outcomes.**

---

## Changelog

### Version 1.0.0 (2026-02-11)
- Initial release
- Core box trading strategy
- 12 layers of protection
- Comprehensive Telegram alerts
- Phased rollout framework
- Performance tracking
- Adaptive learning basics

---

## Credits

Strategy based on "Box Theory" trading concept, enhanced with modern risk management, machine learning concepts, and production-grade engineering.

Built on the AITradeBot_core framework.

---

**Happy Trading! 📈**

Remember: Start small, test thoroughly, and never risk more than you can afford to lose.
