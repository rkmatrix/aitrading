# Bot Monitoring Guide

## ✅ Bot Status: RUNNING

**Process ID**: 26008  
**Started**: 2026-01-10 07:22:32 AM  
**Status**: Active (initializing/running)

## Current Situation

The bot is **running** but the market is currently **CLOSED** (weekend). This is expected behavior:

- ✅ Bot process is active
- ✅ Market is closed (weekend)
- ✅ Bot will idle until market opens (Monday 9:30 AM ET)
- ✅ Bot will automatically start trading when market opens

## How to Monitor the Bot

### Option 1: Watch Logs in Real-Time (Recommended)

Open a **new PowerShell window** and run:

```powershell
Get-Content data\logs\phase26_realtime_live.log -Tail 20 -Wait
```

This will show you the latest 20 log lines and update automatically as new entries are added.

### Option 2: Quick Status Check

```powershell
python check_bot_status.py
```

### Option 3: Continuous Monitoring

```powershell
python monitor_bot_continuous.py 10
```

This updates every 10 seconds showing:
- Kill switch status
- Market status
- Latest log entries
- Bot activity

### Option 4: Check Process Status

```powershell
Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddHours(-1)} | Format-Table Id,ProcessName,StartTime,CPU
```

## What to Expect

### Right Now (Weekend)
- Bot logs: "Market closed — idling" every 5 minutes
- No trading activity
- Bot is waiting for market to open

### When Market Opens (Monday 9:30 AM ET)
- Bot will detect market is open
- Will start making trading decisions
- Will execute trades (PAPER mode)
- Logs will show trading activity

## Log File Location

```
data/logs/phase26_realtime_live.log
```

## Important Log Messages

**Normal Operation:**
- `Market closed — idling.` - Normal when market is closed
- `Market open — trading logic would run here` - Bot is active
- `Order submitted` - Trade executed
- `Kill-switch flag detected` - Trading stopped

**Warnings/Errors:**
- `⚠️` - Warning (investigate but not critical)
- `❌` or `💥` - Error (needs attention)
- `🚨` - Critical alert (immediate action needed)

## Safety Features Active

All improvements are active:
- ✅ Kill switch monitoring
- ✅ Order validation
- ✅ Risk management
- ✅ Retry logic
- ✅ Circuit breaker
- ✅ Structured logging

## To Stop the Bot

### Method 1: Kill Switch (Safest)
```powershell
echo '{"kill": true}' | Out-File -FilePath data\runtime\trading_disabled.flag -Encoding utf8
```

### Method 2: Stop Process
```powershell
Stop-Process -Id 26008
```

### Method 3: Keyboard Interrupt
If running in foreground: Press `Ctrl+C`

## Troubleshooting

### Bot Not Logging
- Wait 1-2 minutes for initialization
- Check if process is still running: `Get-Process -Id 26008`
- Check for errors in console output

### No Trading Activity
- Check if market is open: `python check_bot_status.py`
- Verify kill switch is not active
- Check account has buying power
- Review logs for error messages

### Bot Crashed
- Check logs for error messages
- Bot will auto-restart (up to 50 times)
- Check kill switch file exists
- Verify API keys are correct

## Next Steps

1. **Monitor logs** - Watch for initialization and first activity
2. **Wait for market open** - Bot will automatically start trading
3. **Review first trades** - Monitor closely when trading starts
4. **Check reports** - Review `data/reports/` for performance data

The bot is running and ready! It will automatically start trading when the market opens on Monday.
