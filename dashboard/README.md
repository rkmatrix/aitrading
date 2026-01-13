# AITradingBot Dashboard

Professional-grade trading dashboard with TradingView-level features.

## Quick Start

### Option 1: Run from project root (Recommended)
```bash
# From C:\Projects\trading\AITradeBot_core
python -m dashboard.app
```

### Option 2: Run from dashboard directory
```bash
# From C:\Projects\trading\AITradeBot_core\dashboard
python run.py
```

### Option 3: Use start scripts
**Windows:**
```bash
# From project root
dashboard\start_dashboard.bat
```

**Linux/Mac:**
```bash
# From project root
bash dashboard/start_dashboard.sh
```

## Access Dashboard

Once running, open your browser to:
- **Local**: http://localhost:5000
- **Network**: http://0.0.0.0:5000

## Features

✅ **Ticker Search** - Real-time autocomplete (type "Apple" to find AAPL)
✅ **Ticker Analysis** - Complete company info, charts, performance metrics
✅ **Calendar Widget** - View bot activity by day
✅ **Market Data** - Real-time quotes and historical data via yfinance
✅ **Professional UI** - TradingView-inspired design
✅ **Right Sidebar** - Market status, quick stats, recent activity

## Dependencies

Install required packages:
```bash
pip install -r dashboard/requirements.txt
```

Key dependencies:
- Flask 3.0.0
- yfinance (free market data)
- TradingView Lightweight Charts (via CDN)

## Configuration

Set environment variables (optional):
- `DASHBOARD_SECRET_KEY` - Secret key for sessions (defaults to dev key)
- `PORT` - Port to run on (defaults to 5000)
- `FLASK_ENV` - Environment (development/production)

## Troubleshooting

**Error: "No module named dashboard.app"**
- Make sure you're running from the project root (`C:\Projects\trading\AITradeBot_core`)
- Or use `python run.py` from inside the dashboard directory

**Error: "DASHBOARD_SECRET_KEY not set"**
- This is just a warning in development mode
- Set `DASHBOARD_SECRET_KEY` environment variable for production

**Charts not loading**
- Ensure TradingView Lightweight Charts CDN is accessible
- Check browser console for errors

## API Endpoints

- `/api/market/search` - Search tickers
- `/api/market/ticker/<symbol>/info` - Get ticker info
- `/api/market/ticker/<symbol>/quote` - Get quote
- `/api/ticker/<symbol>/analysis` - Get analysis
- `/api/calendar/activity` - Get calendar activity
- `/api/bot/start` - Start bot
- `/api/bot/stop` - Stop bot

## Development

The dashboard uses:
- **Backend**: Flask with SQLAlchemy
- **Frontend**: Vanilla JavaScript with TradingView Lightweight Charts
- **Real-time**: Flask-SocketIO for WebSocket updates
- **Database**: SQLite (development) / PostgreSQL (production)

## License

Part of the AITradingBot project.
