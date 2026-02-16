# Quick Start Guide - Batch Files

## One-Click Bot Startup

We've created batch files for easy bot startup on Windows.

### start_bot.bat
**Double-click this file to start the bot normally.**

- Starts the trading bot
- Shows all output in the console window
- Keeps window open if bot exits (so you can see errors)
- Press `Ctrl+C` to stop gracefully

### start_bot_debug.bat
**Double-click this file to start the bot in debug mode.**

- Same as `start_bot.bat` but with debug logging enabled
- More verbose output for troubleshooting

## Usage

1. **Double-click** `start_bot.bat` (or `start_bot_debug.bat` for debug mode)
2. The bot will start automatically
3. Watch the console for logs and status updates
4. Press `Ctrl+C` to stop the bot when needed

## Requirements

- Python must be installed and in your PATH
- All dependencies must be installed (`pip install -r requirements.txt`)
- `.env` file must be configured with API keys

## Troubleshooting

If the bot doesn't start:

1. **Check Python installation:**
   ```powershell
   python --version
   ```

2. **Check if you're in the right directory:**
   - The batch file should be in the project root (`AITradeBot_core`)
   - It will automatically change to the correct directory

3. **Check dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Check API keys:**
   - Ensure `.env` file exists with valid Alpaca API credentials

5. **Run manually to see errors:**
   ```powershell
   python runner\phase26_realtime_live.py
   ```

## Customization

You can edit the batch files to:
- Add environment variables
- Change Python interpreter path
- Add pre-startup checks
- Customize the startup message

## Notes

- The console window will stay open after the bot stops (so you can see any error messages)
- To close the window, press any key after the bot stops, or click the X button
- For background operation, you can use Windows Task Scheduler or create a service
