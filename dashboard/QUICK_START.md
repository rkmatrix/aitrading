# Dashboard Quick Start Guide

## Running the Dashboard

### ✅ Recommended: From Project Root
```bash
cd C:\Projects\trading\AITradeBot_core
python -m dashboard.app
```

### ✅ Alternative: From Dashboard Directory
```bash
cd C:\Projects\trading\AITradeBot_core\dashboard
python run.py
```

### ✅ Using Start Scripts
**Windows:**
```bash
dashboard\start_dashboard.bat
```

**Linux/Mac:**
```bash
bash dashboard/start_dashboard.sh
```

## Access Dashboard

Once running, open your browser to:
- **http://localhost:5000**
- **http://127.0.0.1:5000**

## Features

### 🔍 Ticker Search
- Type in the header search box (e.g., "Apple")
- See autocomplete suggestions as you type
- Click to open ticker analysis

### 📊 Ticker Analysis
- Click any ticker in the sidebar to open analysis popup
- View Overview, Chart, Trades, Performance, Signals, and Risk tabs
- Interactive charts with multiple timeframes (1D, 1W, 1M, 3M, 6M, 1Y)

### 📅 Calendar Widget
- View bot activity by day in the right sidebar
- Green = Profitable day, Red = Loss day
- Click any day to see detailed trades and P&L

### 📈 Right Sidebar Widgets
- **Calendar**: Bot activity calendar
- **Market Status**: Current market open/closed status
- **Quick Stats**: Today's P&L, active positions, win rate
- **Recent Activity**: Last 5 trades

## Troubleshooting

### Warning: "DASHBOARD_SECRET_KEY not set"
- This is normal in development mode
- The dashboard will use a default key
- Set `DASHBOARD_SECRET_KEY` environment variable for production

### Error: "can't open file 'run.py'"
- This happens when Flask's auto-reloader tries to restart
- The reloader is now disabled to prevent this issue
- If you need auto-reload, run from project root: `python -m dashboard.app`

### Charts Not Loading
- Ensure you have internet connection (TradingView charts load from CDN)
- Check browser console (F12) for any errors
- Try refreshing the page

### No Market Data
- Market data comes from yfinance (free, no API key needed)
- First request may take a few seconds
- Data is cached for 60 seconds

## Next Steps

1. **Add Tickers**: Click "+ Add" in the ticker list sidebar
2. **Search**: Type ticker symbol or company name in header search
3. **Analyze**: Click any ticker to see detailed analysis
4. **Monitor**: Use calendar and widgets to track bot activity

Enjoy your professional trading dashboard! 🚀
