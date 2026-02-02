# Bot Cloud Deployment Guide - Run Bot Continuously 24/7

This guide shows you how to deploy your **trading bot** (not dashboard) to run continuously in the cloud for **FREE**.

## 🎯 Best Free Options for Running Bot Continuously

1. **Railway** ⭐ (Recommended) - $5 free credit/month, background workers
2. **Render** - Free background workers, easy setup
3. **Fly.io** - Free tier, runs any process
4. **PythonAnywhere** - Free tier, scheduled tasks

---

## 🚀 Option 1: Railway (RECOMMENDED - Best for Bot)

### Why Railway?
- ✅ **Background Workers** - Perfect for long-running bots
- ✅ **$5 free credit/month** - Usually enough for a bot
- ✅ **Stays awake** - No sleep issues
- ✅ **Easy log monitoring** - View logs in dashboard
- ✅ **Auto-restart** - Restarts on crash

### Step-by-Step Instructions:

#### Step 1: Create Railway Account
1. Go to **https://railway.app**
2. Click **"Start a New Project"**
3. Sign up with GitHub (recommended)
4. Verify email if required

#### Step 2: Deploy from GitHub
1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway to access GitHub if prompted
4. Find and select: **`rkmatrix/aitrading`** (or your repo)
5. Click **"Deploy Now"**

#### Step 3: Configure as Background Worker
1. In Railway project, click on the deployed service
2. Go to **"Settings"** tab
3. Under **"Service Type"**, select **"Background Worker"** (not Web Service)
4. This tells Railway it's a long-running process, not a web server

#### Step 4: Set Start Command
1. Still in **"Settings"** tab
2. Find **"Start Command"** section
3. Set it to: `python runner/phase26_realtime_live.py`
4. Save changes

#### Step 5: Configure Environment Variables
1. Go to **"Variables"** tab
2. Click **"New Variable"** and add each:

```
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
ENV=PAPER
```

**Optional variables:**
```
PHASE26_AUTORESTART_MAX=50
PHASE26_AUTORESTART_BACKOFF_SEC=3.0
```

#### Step 6: Deploy
1. Railway will automatically rebuild and deploy
2. Watch the **"Deployments"** tab for build progress
3. Once deployed, check **"Logs"** tab to see bot output

#### Step 7: Monitor Logs
1. Click **"Logs"** tab in Railway dashboard
2. You'll see real-time bot logs
3. Logs are searchable and filterable
4. Railway keeps logs for 7 days (free tier)

**✅ Done!** Your bot is now running 24/7 on Railway!

---

## 🌐 Option 2: Render (Free Background Worker)

### Step 1: Create Render Account
1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with GitHub

### Step 2: Create Background Worker
1. In Render dashboard, click **"New +"**
2. Select **"Background Worker"** (NOT Web Service)
3. Click **"Connect GitHub"** and authorize
4. Find and select: **`rkmatrix/aitrading`**
5. Click **"Connect"**

### Step 3: Configure Worker Settings
Fill in these settings:

- **Name**: `aitradingbot-worker` (or any name)
- **Region**: Choose closest to you
- **Branch**: `main`
- **Root Directory**: (leave blank)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python runner/phase26_realtime_live.py`

### Step 4: Add Environment Variables
Scroll to **"Environment Variables"** section:

```
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
ENV=PAPER
```

### Step 5: Choose Free Plan
1. Scroll to **"Plan"** section
2. Select **"Free"** plan
3. Click **"Create Background Worker"**

### Step 6: Deploy
1. Render will start building
2. Watch build logs
3. Once deployed, check **"Logs"** tab

**✅ Done!** Your bot is running on Render!

**Note**: Render free tier workers may sleep after inactivity. To keep awake, use UptimeRobot (see below).

---

## 🪶 Option 3: Fly.io (Best Free Tier - Stays Awake)

### Step 1: Install Fly CLI
**Windows (PowerShell as Administrator):**
```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

**Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Sign Up
```bash
flyctl auth signup
```

### Step 3: Create fly.toml for Bot
Create `fly.toml` in project root:

```toml
app = "aitradingbot"
primary_region = "iad"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  MODE = "PAPER"
  ENV = "PAPER"

[[services]]
  internal_port = 8080
  protocol = "tcp"
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 1

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true
```

### Step 4: Create Procfile
Create `Procfile` in project root:

```
worker: python runner/phase26_realtime_live.py
```

### Step 5: Deploy
```bash
cd C:\Projects\trading\AITradeBot_core
flyctl launch
```

Follow prompts:
- App name: `aitradingbot` (or your choice)
- Region: Choose closest
- PostgreSQL: **No**
- Redis: **No**
- Deploy now: **Yes**

### Step 6: Set Secrets
```bash
flyctl secrets set APCA_API_KEY_ID=your_key
flyctl secrets set APCA_API_SECRET_KEY=your_secret
flyctl secrets set APCA_API_BASE_URL=https://paper-api.alpaca.markets
flyctl secrets set MODE=PAPER
flyctl secrets set ENV=PAPER
```

### Step 7: View Logs
```bash
flyctl logs
```

**✅ Done!** Your bot is running on Fly.io!

---

## 🐍 Option 4: PythonAnywhere (Simple Free Hosting)

### Step 1: Create Account
1. Go to **https://www.pythonanywhere.com**
2. Click **"Create a Beginner account"** (free)
3. Sign up with email

### Step 2: Clone Repository
1. Login to PythonAnywhere dashboard
2. Go to **"Files"** tab
3. Click **"Open Bash console here"**
4. Run:
   ```bash
   git clone https://github.com/rkmatrix/aitrading.git
   cd aitrading
   ```

### Step 3: Install Dependencies
```bash
pip3.10 install --user -r requirements.txt
```

### Step 4: Set Environment Variables
1. Go to **"Files"** tab
2. Create `.env` file in project root:
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   MODE=PAPER
   ENV=PAPER
   ```

### Step 5: Create Always-On Task
1. Go to **"Tasks"** tab
2. Click **"Create a new task"**
3. Fill in:
   - **Command**: `cd ~/aitrading && python3.10 runner/phase26_realtime_live.py`
   - **Hour**: `*` (every hour)
   - **Minute**: `*` (every minute)
   - **Enabled**: ✅ Checked
4. Click **"Create"**

**Note**: Free tier allows one always-on task. This will keep your bot running.

### Step 6: View Logs
1. Go to **"Tasks"** tab
2. Click on your task
3. View **"Output"** for logs

**✅ Done!** Your bot is running on PythonAnywhere!

---

## 📊 Comparison of Free Options

| Platform | Free Forever? | Stays Awake? | Log Monitoring | Best For |
|----------|---------------|--------------|----------------|----------|
| **Railway** | ✅ $5/month credit | ✅ Yes | ⭐⭐⭐⭐⭐ Excellent | Best overall |
| **Render** | ✅ Yes | ⚠️ May sleep | ⭐⭐⭐⭐ Good | Easy setup |
| **Fly.io** | ✅ Yes | ✅ Yes | ⭐⭐⭐⭐ Good | Docker users |
| **PythonAnywhere** | ✅ Yes | ✅ Yes | ⭐⭐⭐ Basic | Simple hosting |

---

## 🏆 RECOMMENDED: Railway

**Why Railway for Bot Deployment:**
1. ✅ **Background Workers** - Designed for long-running processes
2. ✅ **$5 free credit** - Usually enough for a bot
3. ✅ **Excellent logs** - Real-time, searchable, filterable
4. ✅ **Auto-restart** - Restarts on crash automatically
5. ✅ **No sleep** - Stays awake 24/7
6. ✅ **Easy monitoring** - Built-in dashboard

**Quick Start:**
1. Sign up at railway.app
2. Deploy from GitHub → Select your repo
3. Change service type to **"Background Worker"**
4. Set start command: `python runner/phase26_realtime_live.py`
5. Add environment variables
6. Deploy!

**Total Time: ~5 minutes**
**Total Cost: $0.00** ✅

---

## 📋 Pre-Deployment Checklist

Before deploying:

- [ ] Code is pushed to GitHub
- [ ] You have Alpaca API credentials (paper trading)
- [ ] Tested bot locally (`python runner/phase26_realtime_live.py`)
- [ ] `.env` file has correct values (or ready to set in cloud)
- [ ] `requirements.txt` is up to date

---

## 🔍 Monitoring Your Bot in Cloud

### Railway
- **Logs Tab**: Real-time logs, searchable
- **Metrics Tab**: CPU, memory usage
- **Deployments Tab**: Deployment history

### Render
- **Logs Tab**: Real-time logs
- **Metrics Tab**: Resource usage
- **Events Tab**: Deployment events

### Fly.io
```bash
flyctl logs          # View logs
flyctl status        # Check status
flyctl monitor       # Real-time monitoring
```

### PythonAnywhere
- **Tasks Tab**: View task output
- **Files Tab**: Check log files in `data/logs/`

---

## 🆘 Troubleshooting

### Bot Keeps Crashing
- Check logs for error messages
- Verify environment variables are set
- Ensure `requirements.txt` has all dependencies
- Check if API keys are valid

### Bot Not Trading
- Verify `MODE=PAPER` is set
- Check Alpaca API credentials
- Review logs for market hours messages
- Ensure market is open (9:30 AM - 4:00 PM ET)

### Can't See Logs
- Check if bot is actually running (not sleeping)
- Verify deployment was successful
- Check platform-specific log locations

### High Resource Usage
- Monitor CPU/memory in platform dashboard
- Optimize bot code if needed
- Consider upgrading plan if hitting limits

---

## 💡 Tips for Cloud Bot Success

1. **Monitor Logs Regularly** - Check daily for errors
2. **Set Up Alerts** - Use Telegram alerts for important events
3. **Keep Code Updated** - Push fixes to GitHub, auto-deploys
4. **Watch Resource Usage** - Stay within free tier limits
5. **Backup Important Data** - Export logs/data periodically
6. **Test Locally First** - Always test changes before deploying

---

## 🔄 Keeping Render Worker Awake

If using Render free tier, workers may sleep. Keep awake:

### Using UptimeRobot (Free)
1. Go to **https://uptimerobot.com**
2. Sign up (free, no credit card)
3. Add monitor:
   - Type: HTTP(s)
   - URL: Your Render worker URL (if it has one) OR use a health check endpoint
   - Interval: 5 minutes
4. Done!

**Note**: Background workers on Render may not have HTTP endpoints. In this case, Render workers stay awake as long as they're processing. The weekend optimization we added will help reduce unnecessary checks.

---

## 📞 Next Steps After Deployment

1. **Monitor Logs** - Check bot is running correctly
2. **Verify Trading** - Check if bot is making trades (paper mode)
3. **Set Up Alerts** - Configure Telegram alerts (optional)
4. **Check Daily** - Review logs and metrics daily
5. **Update Code** - Push improvements to GitHub for auto-deploy

---

**Your bot is now running 24/7 in the cloud! 🚀**

**No more local machine needed - monitor everything from the cloud dashboard!**
