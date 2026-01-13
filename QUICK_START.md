# Quick Start Guide - AITradingBot

Get your trading bot up and running in minutes!

## Prerequisites

- Python 3.11 or higher
- Alpaca API account (get free paper trading account at https://alpaca.markets/)

## Installation Steps

### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/AITradingBot.git
cd AITradingBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Alpaca API credentials
# On Windows: notepad .env
# On Mac/Linux: nano .env
```

**Required variables in `.env`:**
```
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
```

### 4. Start the Dashboard

```bash
# From project root
python -m dashboard.app
```

Open http://localhost:5000 in your browser.

### 5. Start the Trading Bot

In a separate terminal:

```bash
python runner/phase26_realtime_live.py
```

## What You'll See

### Dashboard (http://localhost:5000)
- **Metrics Panel**: Equity, P&L, win rate
- **Ticker List**: Manage trading symbols
- **Trades Table**: View all trades
- **Calendar Widget**: See bot activity by day
- **Bot Controls**: Start/stop the bot

### Bot Console
- Market hours check
- Trading decisions
- Order executions
- Risk checks
- Error handling

## First Steps

1. **Add Tickers**: Click "+ Add" in dashboard sidebar, search for tickers (e.g., "Apple")
2. **Monitor**: Watch the dashboard for real-time updates
3. **Review Trades**: Check the trades table for executed orders
4. **Analyze**: Click any ticker to see detailed analysis with charts

## Safety Features

- **Paper Trading**: Default mode uses paper trading (no real money)
- **Kill Switch**: Emergency stop via dashboard or file
- **Risk Limits**: Automatic position sizing and drawdown control
- **Market Hours**: Only trades during market hours

## Troubleshooting

**Dashboard won't start?**
- Check Python version: `python --version` (need 3.11+)
- Install dependencies: `pip install -r requirements.txt`
- Check port 5000 is available

**Bot won't connect?**
- Verify `.env` file exists and has correct API keys
- Check `MODE=PAPER` for paper trading
- Ensure internet connection

**No trades happening?**
- Market might be closed (check market hours)
- Verify tickers are added in dashboard
- Check bot logs for errors

## Next Steps

- Read [START_BOT.md](START_BOT.md) for detailed bot documentation
- Read [DEPLOYMENT.md](DEPLOYMENT.md) to deploy to cloud hosting
- Read [MONITORING_GUIDE.md](MONITORING_GUIDE.md) for monitoring tips

## Getting Help

1. Check the logs: `data/logs/phase26_realtime_live.log`
2. Review dashboard logs in the browser console (F12)
3. Verify environment variables are set correctly
4. Test in PAPER mode before LIVE trading

Happy trading! 🚀
