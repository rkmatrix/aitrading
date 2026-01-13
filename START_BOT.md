# How to Start the AITradingBot

## Main Trading Bot Script

The main script to run the trading bot is:

```bash
python runner/phase26_realtime_live.py
```

Or on Windows PowerShell:
```powershell
python runner\phase26_realtime_live.py
```

## Prerequisites

Before starting the bot, make sure:

1. **Environment Variables are Set**
   - Create a `.env` file in the project root (if not exists)
   - Add your Alpaca API credentials:
     ```
     APCA_API_KEY_ID=your_key_here
     APCA_API_SECRET_KEY=your_secret_here
     APCA_API_BASE_URL=https://paper-api.alpaca.markets
     MODE=PAPER
     ```

2. **Trading Mode**
   - `MODE=PAPER` - Paper trading (recommended for testing)
   - `MODE=LIVE` - Live trading (use with caution!)
   - `MODE=DEMO` - Demo mode (simulated trading)

## Quick Start

### Step 1: Verify Configuration
```bash
python validate_improvements.py
```

This will verify all components are working correctly.

### Step 2: Start the Bot
```bash
python runner/phase26_realtime_live.py
```

The bot will:
- Load environment variables
- Validate API keys
- Connect to Alpaca broker
- Check market hours
- Start trading loop

## What Happens When You Run It

1. **Initialization**
   - Loads configuration from `.env`
   - Validates API keys (fails fast if missing)
   - Initializes broker connection
   - Sets up risk management systems
   - Initializes kill switch monitor

2. **Market Hours Check**
   - Checks if market is open
   - If closed, enters idle mode
   - If open, begins trading loop

3. **Trading Loop**
   - Monitors kill switch
   - Gets market data
   - Makes trading decisions
   - Validates orders
   - Executes trades
   - Monitors positions

4. **Safety Features**
   - Kill switch monitoring (checks every tick)
   - Risk validation on every order
   - Automatic error recovery
   - Circuit breaker for broker failures

## Stopping the Bot

- **Graceful Shutdown**: Press `Ctrl+C` (KeyboardInterrupt)
- **Emergency Stop**: Create kill switch file:
  ```bash
  echo '{"kill": true}' > data/runtime/trading_disabled.flag
  ```

## Alternative Scripts

### Simplified Version (Idle Mode)
```bash
python phase26_realtime_live_PHASE_E_IDLE_NO_SYNTHETIC_FINAL.py
```
This is a minimal version that only idles when market is closed. Use the main script instead.

### Backtest Script
```bash
python runner/phaseG1_backtest_capital.py
```
For backtesting strategies (not live trading).

## Environment Variables

Key environment variables you can set:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODE` | Trading mode (PAPER/LIVE/DEMO) | PAPER |
| `ENV` | Environment name | PAPER_TRADING |
| `APCA_API_KEY_ID` | Alpaca API key | Required |
| `APCA_API_SECRET_KEY` | Alpaca API secret | Required |
| `APCA_API_BASE_URL` | Alpaca API URL | paper-api.alpaca.markets |
| `PHASE26_AUTORESTART_MAX` | Max auto-restarts | 50 |
| `TRADING_KILL_FLAG` | Kill switch file path | data/runtime/trading_disabled.flag |

## Monitoring

### Logs
- Console output: Real-time logging
- File logs: `data/logs/phase26_realtime_live.log`
- Structured logs: JSON format for critical events

### Telegram Alerts
If configured, the bot will send Telegram notifications for:
- Trade executions
- Kill switch activations
- Critical errors
- Market status changes

## Safety Checklist Before Live Trading

- [ ] Tested extensively in PAPER mode
- [ ] Verified kill switch works (`data/runtime/trading_disabled.flag`)
- [ ] Set appropriate position sizes
- [ ] Configured risk limits
- [ ] Tested error recovery
- [ ] Monitored logs for issues
- [ ] Have manual override ready

## Troubleshooting

### Bot won't start
- Check `.env` file exists and has correct API keys
- Run `python validate_improvements.py` to check for issues
- Check Python version (requires Python 3.8+)

### Bot not trading
- Check if market is open (bot idles when market closed)
- Check kill switch file doesn't exist
- Verify account has buying power
- Check logs for error messages

### API Errors
- Verify API keys are correct
- Check API base URL matches your account type
- Ensure account is not blocked

## Example Startup Output

```
2026-01-10 08:00:00 [INFO] Runtime MODE=PAPER ENV=PAPER_TRADING
2026-01-10 08:00:00 [INFO] ✅ API keys validated successfully for PAPER mode
2026-01-10 08:00:01 [INFO] 🔌 Initializing AlpacaClient (PAPER)
2026-01-10 08:00:02 [INFO] ✅ Starting real-time loop…
2026-01-10 08:00:02 [INFO] Market closed — idling.
```

## Important Notes

1. **Always start with PAPER mode** to test before going live
2. **Monitor the first few trades closely**
3. **Keep kill switch file ready** for emergencies
4. **Start with small position sizes**
5. **Review logs regularly** for any issues
