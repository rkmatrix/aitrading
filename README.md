# AITradingBot - Automated Trading System

A professional-grade automated trading bot with AI/ML capabilities, comprehensive risk management, and a TradingView-level dashboard.

## Features

- 🤖 **AI-Powered Trading**: Reinforcement learning and machine learning models for trading decisions
- 📊 **Professional Dashboard**: TradingView-inspired web dashboard with real-time monitoring
- 🛡️ **Risk Management**: Multi-layer risk controls, kill switches, and safety mechanisms
- 📈 **Market Data Integration**: Real-time quotes, historical data, and market analysis
- 🔄 **Smart Order Routing**: Efficient order execution with slippage prediction
- 📅 **Activity Tracking**: Calendar widget showing bot activity by day
- 🔍 **Ticker Analysis**: Complete market analysis with interactive charts

## Quick Start

### Prerequisites

- Python 3.11+
- Alpaca API account (paper or live)
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AITradingBot.git
   cd AITradingBot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Alpaca API credentials
   # See .env.example for all required variables
   ```

4. **Start the dashboard**
   ```bash
   python -m dashboard.app
   ```
   Open http://localhost:5000 in your browser

5. **Start the trading bot**
   ```bash
   python runner/phase26_realtime_live.py
   ```

## Project Structure

```
AITradingBot/
├── ai/                    # Core AI/ML modules
│   ├── execution/         # Order execution and routing
│   ├── risk/              # Risk management
│   ├── policy/            # Trading policies
│   ├── agents/            # Trading agents
│   └── ...
├── runner/                # Execution scripts
│   └── phase26_realtime_live.py  # Main bot runner
├── dashboard/             # Web dashboard
│   ├── app.py             # Flask application
│   ├── api/               # API endpoints
│   ├── static/            # Frontend assets
│   └── templates/         # HTML templates
├── configs/               # Configuration files
├── tools/                 # Utility scripts
├── tests/                 # Test files
├── data/                  # Data files (not in Git)
│   └── policies_src/      # Policy source templates (in Git)
└── models/                # Trained models (not in Git)
```

## Configuration

### Environment Variables

See `.env.example` for all available environment variables. Key variables:

- `APCA_API_KEY_ID` - Alpaca API key (required)
- `APCA_API_SECRET_KEY` - Alpaca secret (required)
- `APCA_API_BASE_URL` - Alpaca API URL
- `MODE` - Trading mode: PAPER, LIVE, or DEMO
- `DASHBOARD_SECRET_KEY` - Dashboard secret key (for production)

### Trading Modes

- **PAPER**: Paper trading (recommended for testing)
- **LIVE**: Real money trading (use with caution!)
- **DEMO**: Simulated trading (no API calls)

## Dashboard Features

- **Ticker Search**: Real-time autocomplete (type "Apple" to find AAPL)
- **Ticker Analysis**: Complete company info, charts, performance metrics
- **Calendar Widget**: View bot activity by day with trade details
- **Market Data**: Real-time quotes and historical data via yfinance
- **Bot Control**: Start/stop bot, monitor status, view logs
- **Portfolio Metrics**: Equity, P&L, win rate, and more
- **Trade History**: Complete trade log with filtering

## Documentation

- **[START_BOT.md](START_BOT.md)** - How to start the trading bot
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deploy to free hosting (Railway, Render, etc.)
- **[GIT_SETUP.md](GIT_SETUP.md)** - Git repository setup guide
- **[MONITORING_GUIDE.md](MONITORING_GUIDE.md)** - Monitoring and troubleshooting
- **[dashboard/README.md](dashboard/README.md)** - Dashboard documentation

## Deployment

The bot and dashboard can be deployed to free hosting platforms:

- **Railway** - Recommended for Python apps
- **Render** - Easy Flask deployment
- **Fly.io** - Docker-based deployment
- **PythonAnywhere** - Free Python hosting

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Security

- ⚠️ **Never commit `.env` file** - It contains API keys
- ✅ Use `.env.example` as a template
- ✅ Set strong `DASHBOARD_SECRET_KEY` for production
- ✅ Rotate API keys if accidentally committed
- ✅ Use environment variables in hosting platforms

## Safety Features

- **Kill Switch**: Emergency stop mechanism
- **Risk Limits**: Position sizing and drawdown controls
- **Order Validation**: Pre-trade checks
- **Circuit Breakers**: Automatic error recovery
- **Market Hours Check**: Only trades during market hours

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- `ai/` - Core trading logic and AI models
- `runner/` - Main execution loops
- `dashboard/` - Web interface
- `configs/` - YAML configuration files
- `tools/` - Utility scripts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and personal use. Use at your own risk.

## Disclaimer

Trading involves risk. This bot is provided as-is without warranty. Always test in paper trading mode before using real money. The authors are not responsible for any financial losses.

## Support

- Check [MONITORING_GUIDE.md](MONITORING_GUIDE.md) for troubleshooting
- Review application logs for errors
- Test in PAPER mode before LIVE trading

## Acknowledgments

Built with:
- Alpaca API for trading
- yfinance for market data
- Flask for the dashboard
- TradingView Lightweight Charts for visualization
