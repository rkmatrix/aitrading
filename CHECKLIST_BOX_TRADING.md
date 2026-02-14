# Box Trading Bot - Pre-Launch Checklist

Use this checklist before starting the bot for the first time.

---

## ✅ Phase 1: Prerequisites (Before Starting)

### Environment Setup
- [ ] Python 3.8+ installed and working
  ```bash
  python --version
  ```
- [ ] Project dependencies installed
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Git repository up to date
  ```bash
  git status
  ```

### API Credentials
- [ ] Alpaca account created (paper trading)
- [ ] API keys generated from Alpaca dashboard
- [ ] Telegram bot created via @BotFather
- [ ] Telegram chat ID obtained

### Configuration Files
- [ ] `.env` file exists in project root
- [ ] `.env` contains all required variables:
  - [ ] `ENV=PAPER_TRADING` (IMPORTANT!)
  - [ ] `APCA_API_KEY_ID=...`
  - [ ] `APCA_API_SECRET_KEY=...`
  - [ ] `APCA_API_BASE_URL=https://paper-api.alpaca.markets`
  - [ ] `TELEGRAM_BOT_TOKEN=...`
  - [ ] `TELEGRAM_CHAT_ID=...`
  - [ ] `TELEGRAM_ENABLED=true`
- [ ] `configs/box_trading.yaml` exists
- [ ] Reviewed configuration settings

---

## ✅ Phase 2: Pre-Flight Checks

### File Verification
- [ ] `ai/strategies/box_trading_strategy.py` exists
- [ ] `runner/box_trading_runner.py` exists
- [ ] `configs/box_trading.yaml` exists
- [ ] `boxTradingbot.bat` exists
- [ ] `README_BOX_TRADING.md` exists
- [ ] `QUICKSTART_BOX_TRADING.md` exists

### Configuration Review
- [ ] Symbols list reviewed (default: SPY, QQQ, AAPL, MSFT)
- [ ] Risk settings appropriate (2% base risk)
- [ ] Max positions set correctly (default: 2)
- [ ] Current phase set to "paper_testing"
- [ ] Telegram alerts enabled
- [ ] Circuit breakers enabled

### Test Telegram
- [ ] Can send test message:
  ```python
  from tools.telegram_alerts import notify
  notify("Box Trading Bot Test", kind="orders")
  ```
- [ ] Received test message in Telegram

---

## ✅ Phase 3: First Launch

### Before Starting
- [ ] Market is open (9:30 AM - 4:00 PM ET)
  - If closed, bot will idle (this is normal)
- [ ] Terminal/command prompt ready
- [ ] Ready to monitor for at least 1 hour

### Start the Bot
- [ ] Double-click `boxTradingbot.bat`
  - OR run: `python runner\box_trading_runner.py`
- [ ] Bot window opened successfully
- [ ] No immediate errors displayed

### Initial Checks (First 5 Minutes)
- [ ] Startup message in console
- [ ] Telegram alert received: "BOX TRADING BOT STARTED"
- [ ] Log file created: `data/logs/box_trading_bot.log`
- [ ] Bot shows "Market open" or "Market closed - idling"

---

## ✅ Phase 4: First Day Monitoring

### During Market Hours
- [ ] Bot is running (window still open)
- [ ] Checking for signals every 30 seconds
- [ ] Logs showing activity
- [ ] If signals found: Telegram alerts received
- [ ] If no signals: This is normal (selective strategy)

### First Trade (When It Happens)
- [ ] Telegram entry alert received
- [ ] Alert contains:
  - [ ] Symbol and action (BUY/SELL)
  - [ ] Entry price and quantity
  - [ ] Stop loss and targets
  - [ ] Box levels
  - [ ] Confidence and reasons
- [ ] Trade appears in log file
- [ ] Position management working

### End of Day (After 4:00 PM ET)
- [ ] All positions closed by 3:55 PM
- [ ] Daily summary alert received
- [ ] Bot still running (will idle until next day)
- [ ] Log file complete for the day

---

## ✅ Phase 5: First Week Review

### Daily Tasks (Each Trading Day)
- [ ] Morning: Check bot is running
- [ ] Morning: Review previous day's summary
- [ ] During day: Monitor Telegram alerts
- [ ] Evening: Review daily summary
- [ ] Evening: Check log file for errors

### End of Week
- [ ] Total trades executed: _____
- [ ] Win rate: _____% (aim for >50%)
- [ ] Any critical errors? (Yes/No) _____
- [ ] Circuit breakers triggered? (Yes/No) _____
- [ ] Telegram alerts working? (Yes/No) _____

### Week 1 Goals
- [ ] Bot runs without crashes ✓
- [ ] Telegram alerts working ✓
- [ ] At least 5 trades executed ✓
- [ ] Understand strategy behavior ✓
- [ ] No major bugs found ✓

---

## ✅ Phase 6: Paper Trading (Weeks 1-4)

### Weekly Checks
- [ ] Week 1 completed successfully
- [ ] Week 2 completed successfully
- [ ] Week 3 completed successfully
- [ ] Week 4 completed successfully

### Performance Tracking
- [ ] Total trades: _____ (need 50+)
- [ ] Win rate: _____% (need >55%)
- [ ] Profit factor: _____ (need >1.4)
- [ ] Max drawdown: _____% (need <8%)

### Before Validation
- [ ] Minimum 50 trades completed
- [ ] At least 28 days of testing
- [ ] Understand why wins won
- [ ] Understand why losses lost
- [ ] Comfortable with strategy

---

## ✅ Phase 7: Validation (After 4+ Weeks)

### Run Validator
- [ ] Execute validation:
  ```bash
  python tools\validate_box_trading.py
  ```
- [ ] Validation report generated
- [ ] Review all metrics carefully

### Validation Results
- [ ] Total trades: _____ (✓ >50)
- [ ] Win rate: _____% (✓ >55%)
- [ ] Profit factor: _____ (✓ >1.4)
- [ ] Max drawdown: _____% (✓ <8%)
- [ ] Duration: _____ days (✓ >28)

### Pass/Fail
- [ ] **PASSED** - Ready for Live Phase 1
- [ ] **FAILED** - Continue paper trading

If FAILED:
- [ ] Identified failure reasons
- [ ] Plan to address issues
- [ ] Extended paper trading period

---

## ✅ Phase 8: Going Live (Only If Validation Passed)

### Pre-Live Checklist
- [ ] Validation PASSED ✓
- [ ] Reviewed ALL documentation ✓
- [ ] Comfortable with strategy ✓
- [ ] Capital allocated ($500-1000 recommended)
- [ ] Emotionally prepared for losses
- [ ] Ready to monitor intensely

### Configuration Changes
- [ ] Updated `.env`: `ENV=LIVE`
- [ ] Updated config: `current_phase: "live_phase1"`
- [ ] Set `max_position_size_dollars: 100`
- [ ] Set `max_positions: 1`
- [ ] Backed up current configuration

### Live Launch
- [ ] DOUBLE-CHECKED: ENV=LIVE in .env
- [ ] Started bot during market hours
- [ ] Received startup alert
- [ ] Bot running in LIVE mode confirmed
- [ ] Ready to monitor every trade

### First Live Trade
- [ ] Entry alert received
- [ ] Verified in Alpaca dashboard
- [ ] Position size correct ($100 max)
- [ ] Stop loss in place
- [ ] Monitoring closely

---

## ✅ Phase 9: Live Phase 1 (Weeks 1-2)

### Daily Monitoring
- [ ] Day 1: Monitor every trade
- [ ] Day 2: Monitor every trade
- [ ] Day 3: Monitor every trade
- [ ] Day 4: Monitor every trade
- [ ] Day 5: Monitor every trade
- [ ] Week 1 complete: Review performance
- [ ] Week 2 complete: Review performance

### Live Phase 1 Goals
- [ ] 10+ live trades completed
- [ ] Win rate >50%
- [ ] No execution errors
- [ ] Slippage acceptable
- [ ] Emotional control maintained
- [ ] Ready for Phase 2

---

## ✅ Phase 10: Ongoing Operations

### Weekly Tasks
- [ ] Review performance metrics
- [ ] Analyze losing trades
- [ ] Check for patterns
- [ ] Adjust blacklists if needed
- [ ] Update configuration if needed

### Monthly Tasks
- [ ] Calculate Sharpe ratio
- [ ] Compare vs benchmark
- [ ] Deep dive analysis
- [ ] Parameter optimization
- [ ] Documentation updates

### Quarterly Tasks
- [ ] Full strategy review
- [ ] Market condition analysis
- [ ] Consider symbol changes
- [ ] Evaluate phase progression
- [ ] Plan improvements

---

## 🚨 Emergency Procedures

### If Bot Crashes
1. [ ] Check log file for errors
2. [ ] Check open positions in Alpaca
3. [ ] Close positions manually if needed
4. [ ] Fix identified issue
5. [ ] Restart bot
6. [ ] Monitor closely

### If Large Loss
1. [ ] Verify circuit breaker didn't trigger
2. [ ] Check if settings correct
3. [ ] Review trade log
4. [ ] Pause bot if necessary
5. [ ] Analyze root cause
6. [ ] Adjust strategy if needed

### If Win Rate Drops Below 45%
1. [ ] Pause bot immediately
2. [ ] Analyze last 20 trades
3. [ ] Check market conditions
4. [ ] Review configuration
5. [ ] Return to paper trading if needed

---

## 📝 Notes Section

### Important Dates
- Paper trading started: __________
- First trade: __________
- Validation passed: __________
- Live trading started: __________

### Performance Milestones
- First profitable day: __________
- First profitable week: __________
- 50 trades completed: __________
- 55% win rate achieved: __________

### Issues Encountered
_Use this space to document any issues and resolutions:_

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

### Configuration Changes
_Document any changes made and why:_

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## ✅ Final Verification

Before considering the implementation complete:

- [ ] All files created and verified
- [ ] Bot tested in paper trading
- [ ] Documentation read and understood
- [ ] Telegram alerts working
- [ ] Validation tool tested
- [ ] Emergency procedures understood
- [ ] Risk disclaimers acknowledged
- [ ] Ready to commit to 4+ week testing period

---

**Signature:** _________________ **Date:** _________

By checking all boxes and signing above, I confirm:
1. I understand this is a trading bot that can lose money
2. I will start with paper trading for minimum 4 weeks
3. I will only go live after validation passes
4. I will start live with tiny positions ($100 max)
5. I take full responsibility for all trading outcomes
6. I have read and understood all documentation

---

**GOOD LUCK! 🚀**

Print this checklist and check off items as you complete them.
