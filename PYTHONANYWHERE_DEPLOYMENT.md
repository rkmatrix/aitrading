# PythonAnywhere Deployment Guide - Step by Step

Complete guide to deploy your trading bot on PythonAnywhere **FREE** tier and run it continuously.

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ GitHub account with your bot repository pushed
- ✅ PythonAnywhere account (we'll create this)
- ✅ Alpaca API credentials (paper trading)
- ✅ About 15-20 minutes

---

## Step 1: Create PythonAnywhere Account

1. **Go to PythonAnywhere**: https://www.pythonanywhere.com
2. **Click** "Create a Beginner account" (free tier)
3. **Sign up** with:
   - Email address
   - Username (choose carefully - this will be in your URL)
   - Password
4. **Verify your email** (check inbox for verification link)
5. **Login** to your account

**Note**: Free tier includes:
- ✅ 1 always-on task (perfect for your bot!)
- ✅ 512 MB disk space
- ✅ Python 3.10 and 3.11
- ✅ Web app hosting
- ✅ Scheduled tasks

---

## Step 2: Clone Your Repository

1. **Login** to PythonAnywhere dashboard
2. Click on **"Files"** tab (top navigation)
3. Click **"Open Bash console here"** button
   - This opens a terminal in your home directory
4. **Clone your repository**:
   ```bash
   git clone https://github.com/rkmatrix/aitrading.git
   ```
   (Replace `rkmatrix/aitrading` with your actual GitHub repo if different)

5. **Navigate into the project**:
   ```bash
   cd aitrading
   ```

6. **Verify files are there**:
   ```bash
   ls -la
   ```
   You should see files like `runner/`, `ai/`, `configs/`, etc.

---

## Step 3: Install Python Dependencies

1. **Still in the Bash console**, make sure you're in the project directory:
   ```bash
   pwd
   # Should show: /home/yourusername/aitrading
   ```

2. **Install dependencies**:
   ```bash
   pip3.10 install --user -r requirements.txt
   ```
   
   **Note**: 
   - `pip3.10` uses Python 3.10 (or use `pip3.11` for Python 3.11)
   - `--user` installs packages in your user directory (required on free tier)
   - This may take 2-5 minutes

3. **Wait for installation** to complete. You'll see output like:
   ```
   Collecting pandas>=2.2.0
   Downloading pandas-2.2.0...
   ...
   Successfully installed pandas-2.2.0 ...
   ```

4. **Verify installation** (optional):
   ```bash
   python3.10 -c "import pandas; print('Pandas installed:', pandas.__version__)"
   ```

---

## Step 4: Set Up Environment Variables

### Option A: Create .env File (Recommended)

1. **In the Bash console**, create `.env` file:
   ```bash
   cd ~/aitrading
   nano .env
   ```

2. **Add your environment variables**:
   ```
   APCA_API_KEY_ID=your_alpaca_api_key_here
   APCA_API_SECRET_KEY=your_alpaca_secret_key_here
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   MODE=PAPER
   ENV=PAPER
   PHASE26_AUTORESTART_MAX=50
   PHASE26_AUTORESTART_BACKOFF_SEC=3.0
   ```

3. **Save and exit**:
   - Press `Ctrl + X`
   - Press `Y` to confirm
   - Press `Enter` to save

### Option B: Set in PythonAnywhere Dashboard

1. Go to **"Tasks"** tab
2. When creating task, you can set environment variables there (see Step 5)

---

## Step 5: Create Always-On Task

This is the key step - PythonAnywhere free tier allows **1 always-on task** that runs continuously.

1. **Go to "Tasks" tab** (top navigation in PythonAnywhere dashboard)

2. **Click "Create a new task"** button

3. **Fill in the task form**:

   **Command**:
   ```
   cd ~/aitrading && python3.10 runner/phase26_realtime_live.py
   ```
   
   **Note**: 
   - `cd ~/aitrading` ensures we're in the right directory
   - `python3.10` uses Python 3.10 (or `python3.11` if you prefer)
   - Replace `yourusername` with your actual PythonAnywhere username if needed

   **Hour**: `*` (every hour - but task stays running)
   
   **Minute**: `*` (every minute - but task stays running)
   
   **Enabled**: ✅ **Check this box** (very important!)

   **Description** (optional):
   ```
   AITradingBot - Continuous Trading Bot
   ```

4. **Click "Create"** button

5. **Task will start immediately** - you should see it in the tasks list

---

## Step 6: Verify Bot is Running

1. **Check task status**:
   - Go to **"Tasks"** tab
   - Find your task in the list
   - Status should show **"Running"** or **"Enabled"**

2. **View task output**:
   - Click on your task name
   - Click **"Output"** tab
   - You should see bot logs like:
     ```
     INFO:Phase26RealtimeUltra:Phase26RealtimeUltra: guardian_enabled=True...
     INFO:Phase26RealtimeUltra:Market closed — idling.
     ```

3. **Check for errors**:
   - If you see errors, check the output
   - Common issues:
     - Missing dependencies → Re-run `pip3.10 install --user -r requirements.txt`
     - Wrong Python version → Change to `python3.11` in task command
     - Missing environment variables → Check `.env` file

---

## Step 7: Monitor Logs

### Method 1: View Task Output (Real-time)

1. Go to **"Tasks"** tab
2. Click on your task
3. Click **"Output"** tab
4. Logs update in real-time
5. **Refresh** page to see latest logs

### Method 2: View Log Files

1. Go to **"Files"** tab
2. Navigate to: `~/aitrading/data/logs/`
3. Open `phase26_realtime_live.log`
4. View logs in PythonAnywhere file editor

### Method 3: Download Logs

1. Go to **"Files"** tab
2. Navigate to log file
3. Click **"Download"** button
4. View in your local text editor

---

## Step 8: Keep Bot Running

### Automatic Restart

Your bot has built-in auto-restart:
- If bot crashes, it will restart automatically (up to 50 times)
- Check `PHASE26_AUTORESTART_MAX` in `.env` to adjust

### Manual Restart

If you need to restart manually:

1. Go to **"Tasks"** tab
2. Find your task
3. Click **"Stop"** button (if running)
4. Click **"Run"** button to start again

### Update Code

When you push updates to GitHub:

1. **In Bash console**:
   ```bash
   cd ~/aitrading
   git pull origin main
   ```

2. **Restart task**:
   - Go to **"Tasks"** tab
   - Stop and start your task

3. **Or** task will pick up changes on next restart

---

## 🔧 Troubleshooting

### Problem: Task Won't Start

**Solution**:
1. Check task command is correct
2. Verify Python version (`python3.10 --version`)
3. Check if dependencies are installed
4. View task output for error messages

### Problem: "Module not found" Error

**Solution**:
```bash
cd ~/aitrading
pip3.10 install --user -r requirements.txt
```

### Problem: Bot Keeps Crashing

**Solution**:
1. Check task output for error messages
2. Verify environment variables are set correctly
3. Check Alpaca API credentials
4. Ensure `.env` file exists and is readable

### Problem: Can't See Logs

**Solution**:
1. Check task is actually running (not stopped)
2. Wait a few minutes for logs to appear
3. Check `data/logs/` directory exists
4. Verify bot has write permissions

### Problem: "Permission denied" Error

**Solution**:
```bash
cd ~/aitrading
chmod +x runner/phase26_realtime_live.py
```

### Problem: Bot Not Trading

**Solution**:
1. Check `MODE=PAPER` is set
2. Verify Alpaca API credentials are correct
3. Check if market is open (9:30 AM - 4:00 PM ET)
4. Review logs for specific error messages

---

## 📊 Monitoring Your Bot

### Daily Checks

1. **Morning**: Check if bot started correctly
2. **During Market Hours**: Verify trading activity
3. **Evening**: Review daily performance

### What to Monitor

- ✅ Bot is running (not stopped)
- ✅ No error messages in logs
- ✅ Trading activity (if market is open)
- ✅ Resource usage (stay within free tier limits)

### PythonAnywhere Dashboard

- **Tasks Tab**: See task status and output
- **Files Tab**: Browse project files and logs
- **Web Tab**: (Optional) Host dashboard if needed
- **Consoles Tab**: Open Bash/Python consoles

---

## 💡 Tips for Success

1. **Keep Code Updated**:
   ```bash
   cd ~/aitrading
   git pull origin main
   ```

2. **Monitor Daily**: Check task output regularly

3. **Backup Logs**: Download important logs periodically

4. **Stay Within Limits**: Free tier has:
   - 512 MB disk space
   - 1 always-on task
   - CPU time limits (usually fine for a bot)

5. **Use Paper Trading**: Safer for testing, same resources

---

## 🎯 Quick Reference Commands

### Start/Stop Task
- **Start**: Tasks tab → Click "Run"
- **Stop**: Tasks tab → Click "Stop"
- **Restart**: Stop then Start

### View Logs
```bash
cd ~/aitrading
tail -f data/logs/phase26_realtime_live.log
```

### Update Code
```bash
cd ~/aitrading
git pull origin main
```

### Check Bot Status
- Tasks tab → View task status
- Click task → View output

---

## ✅ Deployment Checklist

Before considering deployment complete:

- [ ] PythonAnywhere account created
- [ ] Repository cloned successfully
- [ ] Dependencies installed (`pip3.10 install --user -r requirements.txt`)
- [ ] `.env` file created with API keys
- [ ] Always-on task created and enabled
- [ ] Task is running (check status)
- [ ] Logs are visible in task output
- [ ] Bot is idling (if market closed) or trading (if market open)
- [ ] No errors in logs

---

## 🚀 You're Done!

Your bot is now running continuously on PythonAnywhere!

**What happens next:**
- ✅ Bot runs 24/7
- ✅ Auto-restarts on crash
- ✅ Logs available in dashboard
- ✅ No local machine needed

**Monitor your bot:**
- Go to PythonAnywhere dashboard
- Click "Tasks" tab
- View your task output
- Check logs daily

---

## 📞 Need Help?

1. **Check task output** for error messages
2. **Review logs** in `data/logs/` directory
3. **PythonAnywhere help**: https://help.pythonanywhere.com
4. **Common issues**: See Troubleshooting section above

---

**Your bot is now running in the cloud! 🎉**

**No more local machine needed - monitor everything from PythonAnywhere dashboard!**
