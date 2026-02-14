# Box Trading Bot - Quick Start Guide

## 🚀 Start Here

This is your **5-minute quick start** to get the Box Trading Bot running.

---

## Step 1: Verify Prerequisites

✅ Check you have:
- [ ] Python 3.8+ installed
- [ ] Alpaca account (paper trading)
- [ ] Telegram bot setup
- [ ] `.env` file configured

**Quick Test:**
```bash
python --version
# Should show Python 3.8 or higher
```

---

## Step 2: Configure Environment

**Edit your `.env` file:**
```env
# IMPORTANT: Start with PAPER_TRADING!
ENV=PAPER_TRADING

# Alpaca API Keys
APCA_API_KEY_ID=your_alpaca_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# Telegram (for alerts)
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ENABLED=true
```

---

## Step 3: Review Configuration

**Check `configs/box_trading.yaml`:**

The defaults are conservative and safe. Key settings:
```yaml
symbols: [SPY, QQQ, AAPL, MSFT]  # Liquid symbols
max_positions: 2                  # Conservative
base_risk_per_trade: 0.02        # 2% risk per trade
current_phase: "paper_testing"   # Start here!
```

**You can use defaults as-is** for initial testing.

---

## Step 4: Start the Bot

**Double-click:**
```
boxTradingbot.bat
```

**Or run manually:**
```bash
python runner\box_trading_runner.py
```

---

## Step 5: What to Expect

### First 30 Minutes
```
✅ Bot starts
✅ Telegram alert: "BOX TRADING BOT STARTED"
✅ Checks if market is open
   - If closed: Bot will idle (this is normal)
   - If open: Bot will scan for signals
```

### During Trading Hours (9:30 AM - 3:45 PM ET)
```
📊 Bot scans symbols every 30 seconds
🎯 Executes trades when conditions met
📱 Sends Telegram alert for every trade
⏰ Closes all positions by 3:55 PM ET
```

### End of Day
```
📊 Sends daily summary via Telegram
📝 Logs all activity to: data/logs/box_trading_bot.log
```

---

## Step 6: Monitor Performance

### Check Telegram Alerts

You'll receive:
- ✅ Trade entry alerts (with full details)
- ✅ Trade exit alerts (with P&L)
- ⚠️ Warning alerts (circuit breakers)
- 📊 Daily summary (4:05 PM ET)

### Review Logs

```bash
# View recent activity
tail -f data/logs/box_trading_bot.log

# Windows equivalent
powershell Get-Content data/logs/box_trading_bot.log -Wait
```

---

## Step 7: After 4+ Weeks of Paper Trading

**Run validation:**
```bash
python tools/validate_box_trading.py
```

**If validation passes:**
1. Review README_BOX_TRADING.md thoroughly
2. Understand all metrics
3. Move to Live Phase 1 (tiny positions)
4. Change `ENV=LIVE` in .env
5. Update `current_phase: "live_phase1"` in config

---

## Common First-Time Issues

### "No trades being executed"
✅ **Normal if:**
- Market is closed (weekends, holidays)
- Before 10:00 AM ET (first 30 min avoided)
- After 3:45 PM ET (no new trades)
- Market is trending (not ranging)
- No valid signals (all confirmations must pass)

⚠️ **Check:**
- Is market open? (9:30 AM - 4:00 PM ET)
- Review logs for "skipped" messages
- Trending markets = fewer signals (by design)

### "Bot stops immediately"
⚠️ **Check:**
- Python version (need 3.8+)
- .env file exists
- API keys are correct
- Config file exists
- Review log file for errors

### "No Telegram alerts"
⚠️ **Check:**
- TELEGRAM_BOT_TOKEN in .env
- TELEGRAM_CHAT_ID in .env
- TELEGRAM_ENABLED=true in .env
- Bot actually started (check window)

---

## Emergency Stops

### Stop the Bot
Press `Ctrl+C` in the command window

The bot will:
1. Close all open positions
2. Send final summary
3. Shutdown safely

### Force Kill (if frozen)
Press `Ctrl+C` twice rapidly

---

## Next Steps

1. **Run for 4+ weeks** in paper trading
2. **Accumulate 50+ trades**
3. **Run validation** (see Step 7)
4. **Read full README** before going live
5. **Start tiny** ($100 positions) in live

---

## Getting Help

1. Read `README_BOX_TRADING.md` (comprehensive guide)
2. Check log file: `data/logs/box_trading_bot.log`
3. Review configuration: `configs/box_trading.yaml`
4. Examine the plan file for strategy details

---

## Safety Reminders

⚠️ **IMPORTANT:**
- Start with PAPER_TRADING (no real money)
- Test for minimum 4 weeks
- Achieve 55%+ win rate before live
- Start live with $100 max positions
- Never trade money you can't afford to lose

✅ **You're Protected By:**
- 12 circuit breakers
- Daily loss limits (5%)
- Consecutive loss protection
- Automatic position closing
- No overnight risk (closes by 3:55 PM)

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start bot | `boxTradingbot.bat` or `python runner\box_trading_runner.py` |
| Stop bot | Press `Ctrl+C` |
| View logs | `data\logs\box_trading_bot.log` |
| Validation | `python tools\validate_box_trading.py` |
| Config | `configs\box_trading.yaml` |
| Full docs | `README_BOX_TRADING.md` |

---

**You're Ready to Start! 🎉**

Run `boxTradingbot.bat` and watch the magic happen.

Remember: **Patience is key.** The bot is designed to be selective and only trade when conditions are optimal.

---

**Questions?** Read the full README_BOX_TRADING.md for detailed explanations.
