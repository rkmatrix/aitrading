# Completely FREE Deployment Guide - No Credit Card Required

This guide shows you how to deploy your bot **100% FREE** with no credit card, no trial periods, and no payment required.

## 🎯 Best Completely Free Options

1. **Render** ⭐ (Recommended) - Free forever, just sleeps after inactivity
2. **Fly.io** - Generous free tier, stays awake
3. **Replit** - Free tier, good for Python apps
4. **PythonAnywhere** - Free tier, simple setup

---

## 🌐 Option 1: Render (RECOMMENDED - Easiest Free Option)

### ✅ Pros:
- **100% FREE forever** - No credit card required
- Easy GitHub integration
- Automatic HTTPS
- Good documentation

### ⚠️ Cons:
- Apps sleep after 15 minutes of inactivity
- **Solution**: Use UptimeRobot (free) to keep it awake

### Step-by-Step Instructions:

#### Step 1: Create Render Account
1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with **GitHub** (recommended - easiest)
4. Authorize Render to access your GitHub

#### Step 2: Create Web Service
1. In Render dashboard, click **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see "Connect a repository" - click **"Connect account"** if needed
4. Find and select: **`rkmatrix/aitrading`**
5. Click **"Connect"**

#### Step 3: Configure Service
Fill in these exact settings:

**Basic Settings:**
- **Name**: `aitradingbot-dashboard` (or any name you like)
- **Region**: Choose closest to you
  - `Oregon (US West)` - Good for US
  - `Frankfurt (EU)` - Good for Europe
  - `Singapore (Asia Pacific)` - Good for Asia
- **Branch**: `main` (should auto-detect)
- **Root Directory**: (leave **BLANK** - empty)

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m dashboard.app`

#### Step 4: Add Environment Variables
Scroll down to **"Environment Variables"** section:

Click **"Add Environment Variable"** and add each one:

```
Key: APCA_API_KEY_ID
Value: your_alpaca_api_key_here
```

```
Key: APCA_API_SECRET_KEY
Value: your_alpaca_secret_key_here
```

```
Key: APCA_API_BASE_URL
Value: https://paper-api.alpaca.markets
```

```
Key: MODE
Value: PAPER
```

```
Key: DASHBOARD_SECRET_KEY
Value: generate-a-random-secret-key-here
```

**To generate DASHBOARD_SECRET_KEY:**
- Run: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Copy the output and paste as value

```
Key: PORT
Value: 5000
```

```
Key: FLASK_ENV
Value: production
```

#### Step 5: Choose FREE Plan
1. Scroll down to **"Plan"** section
2. Select **"Free"** plan (should be selected by default)
3. **DO NOT** select "Starter" or any paid plan

#### Step 6: Create and Deploy
1. Scroll to bottom
2. Click **"Create Web Service"**
3. Render will start building your app
4. Watch the build logs - wait for "Your service is live"
5. Your URL will be: `https://aitradingbot-dashboard.onrender.com` (or similar)

#### Step 7: Keep It Awake (IMPORTANT!)
Render free tier apps sleep after 15 minutes of inactivity. Keep it awake:

**Using UptimeRobot (Free):**
1. Go to **https://uptimerobot.com**
2. Sign up for free account (no credit card)
3. Click **"Add New Monitor"**
4. Configure:
   - **Monitor Type**: `HTTP(s)`
   - **Friendly Name**: `AITradingBot Keep Alive`
   - **URL**: Your Render URL (e.g., `https://aitradingbot-dashboard.onrender.com`)
   - **Monitoring Interval**: `5 minutes`
5. Click **"Create Monitor"**

UptimeRobot will ping your app every 5 minutes, keeping it awake 24/7!

**✅ Done!** Your bot is now running FREE on Render.

---

## 🪶 Option 2: Fly.io (Best Free Tier - Stays Awake)

### ✅ Pros:
- **100% FREE** - Generous free tier
- **Stays awake** - No sleep issues
- Good performance
- Docker-based

### ⚠️ Cons:
- Requires CLI installation
- Slightly more setup

### Step-by-Step Instructions:

#### Step 1: Install Fly CLI
**Windows (PowerShell as Administrator):**
```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

**Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

#### Step 2: Sign Up
1. Run: `flyctl auth signup`
2. Follow prompts to create account
3. Verify your email

#### Step 3: Launch App
1. Open terminal/PowerShell in your project directory:
   ```powershell
   cd C:\Projects\trading\AITradeBot_core
   ```

2. Launch app:
   ```powershell
   flyctl launch
   ```

3. Answer prompts:
   - **App name**: `aitradingbot` (or choose your own - must be unique)
   - **Region**: Choose closest (e.g., `iad` for Virginia, `sjc` for California)
   - **PostgreSQL database**: Answer **`n`** (we use SQLite)
   - **Redis**: Answer **`n`**
   - **Deploy now**: Answer **`y`**

#### Step 4: Set Secrets (Environment Variables)
```powershell
flyctl secrets set APCA_API_KEY_ID=your_alpaca_key_here
flyctl secrets set APCA_API_SECRET_KEY=your_alpaca_secret_here
flyctl secrets set APCA_API_BASE_URL=https://paper-api.alpaca.markets
flyctl secrets set MODE=PAPER
flyctl secrets set DASHBOARD_SECRET_KEY=your-random-secret-key-here
flyctl secrets set FLASK_ENV=production
```

#### Step 5: Create fly.toml Configuration
Fly.io should have created `fly.toml`. Edit it to ensure correct settings:

```toml
app = "aitradingbot"
primary_region = "iad"

[build]
  dockerfile = "dashboard/Dockerfile"

[env]
  PORT = "5000"
  FLASK_ENV = "production"

[[services]]
  internal_port = 5000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

#### Step 6: Deploy
```powershell
flyctl deploy
```

#### Step 7: Open App
```powershell
flyctl open
```

**✅ Done!** Your bot is running FREE on Fly.io and stays awake!

---

## 🐍 Option 3: PythonAnywhere (Simple Free Hosting)

### ✅ Pros:
- **100% FREE** - Free tier available
- Simple Python hosting
- No Docker needed

### ⚠️ Cons:
- Free tier has limitations
- Manual setup required

### Step-by-Step Instructions:

#### Step 1: Create Account
1. Go to **https://www.pythonanywhere.com**
2. Click **"Create a Beginner account"** (free)
3. Sign up with email

#### Step 2: Clone Repository
1. Login to PythonAnywhere dashboard
2. Go to **"Files"** tab
3. Click **"Open Bash console here"**
4. Run:
   ```bash
   git clone https://github.com/rkmatrix/aitrading.git
   cd aitrading
   ```

#### Step 3: Install Dependencies
In the Bash console:
```bash
pip3.10 install --user -r requirements.txt
```

#### Step 4: Create Web App
1. Go to **"Web"** tab
2. Click **"Add a new web app"**
3. Choose **"Flask"**
4. Python version: **3.10**
5. Path: `/home/yourusername/aitrading/dashboard/app.py`
   (Replace `yourusername` with your PythonAnywhere username)
6. Click **"Next"** → **"Next"** → **"Finish"**

#### Step 5: Configure WSGI File
1. Click **"WSGI configuration file"** link
2. Replace content with:
   ```python
   import sys
   path = '/home/yourusername/aitrading'
   if path not in sys.path:
       sys.path.insert(0, path)
   
   from dashboard.app import app as application
   ```
   (Replace `yourusername` with your actual username)

#### Step 6: Set Environment Variables
1. Go to **"Web"** tab
2. Click **"Environment variables"**
3. Add each variable:
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   MODE=PAPER
   DASHBOARD_SECRET_KEY=your-secret-key
   FLASK_ENV=production
   ```

#### Step 7: Initialize Database
In Bash console:
```bash
cd aitrading
python3.10 dashboard/init_db.py
```

#### Step 8: Reload Web App
1. Go to **"Web"** tab
2. Click **"Reload"** button
3. Your app URL: `https://yourusername.pythonanywhere.com`

**✅ Done!** Your bot is running FREE on PythonAnywhere.

---

## 🎨 Option 4: Replit (Easy Free Hosting)

### ✅ Pros:
- **100% FREE** - Free tier available
- Easy GitHub import
- Built-in code editor

### Step-by-Step Instructions:

#### Step 1: Create Account
1. Go to **https://replit.com**
2. Sign up with GitHub

#### Step 2: Import from GitHub
1. Click **"Create Repl"**
2. Click **"Import from GitHub"**
3. Enter: `rkmatrix/aitrading`
4. Click **"Import"**

#### Step 3: Configure
1. Replit will detect Python automatically
2. Go to **"Secrets"** tab (lock icon)
3. Add environment variables:
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   MODE=PAPER
   DASHBOARD_SECRET_KEY=your-secret-key
   PORT=5000
   FLASK_ENV=production
   ```

#### Step 4: Install Dependencies
In Replit shell:
```bash
pip install -r requirements.txt
```

#### Step 5: Run
1. Create `.replit` file:
   ```
   run = "python -m dashboard.app"
   ```
2. Click **"Run"** button
3. Replit provides a URL like: `https://your-repl-name.your-username.repl.co`

**✅ Done!** Your bot is running FREE on Replit.

---

## 📊 Comparison of Free Options

| Platform | Free Forever? | Stays Awake? | Ease of Setup | Best For |
|----------|---------------|--------------|--------------|----------|
| **Render** | ✅ Yes | ⚠️ Sleeps (use UptimeRobot) | ⭐⭐⭐⭐⭐ Very Easy | Beginners |
| **Fly.io** | ✅ Yes | ✅ Yes | ⭐⭐⭐⭐ Easy | Best performance |
| **PythonAnywhere** | ✅ Yes | ✅ Yes | ⭐⭐⭐ Medium | Simple hosting |
| **Replit** | ✅ Yes | ⚠️ Sleeps | ⭐⭐⭐⭐ Easy | Development |

---

## 🏆 RECOMMENDED: Render + UptimeRobot

**Why this combination:**
1. ✅ **100% FREE** - No credit card, no payment
2. ✅ **Easy setup** - Just connect GitHub
3. ✅ **Stays awake** - With UptimeRobot (also free)
4. ✅ **Automatic HTTPS** - Secure by default
5. ✅ **Good performance** - Reliable hosting

**Total Cost: $0.00** 🎉

---

## 🔄 Keeping Render Awake (Detailed)

### Method 1: UptimeRobot (Recommended)
1. Sign up: https://uptimerobot.com (free, no credit card)
2. Add monitor:
   - Type: HTTP(s)
   - URL: Your Render URL
   - Interval: 5 minutes
3. Done! Your app stays awake 24/7

### Method 2: Cron-Job.org (Alternative)
1. Sign up: https://cron-job.org (free)
2. Create cron job:
   - URL: Your Render URL
   - Schedule: Every 5 minutes
3. Done!

### Method 3: Python Script (Self-hosted)
If you have another server/computer:
```python
import requests
import time

while True:
    requests.get("https://your-app.onrender.com")
    time.sleep(300)  # 5 minutes
```

---

## 📋 Pre-Deployment Checklist

Before deploying:

- [ ] Code is on GitHub (`rkmatrix/aitrading`)
- [ ] You have Alpaca API credentials (paper trading)
- [ ] Generated `DASHBOARD_SECRET_KEY`
- [ ] Tested dashboard locally

---

## 🆘 Troubleshooting

### Render: App Keeps Sleeping
- **Solution**: Set up UptimeRobot (see above)
- Check UptimeRobot is pinging every 5 minutes

### Build Fails
- Check build logs in Render dashboard
- Ensure `requirements.txt` exists
- Verify Python version compatibility

### App Crashes
- Check runtime logs
- Verify all environment variables are set
- Run `python dashboard/init_db.py` if database errors

### Can't Access Dashboard
- Wait for deployment to complete (can take 2-5 minutes)
- Check if app is awake (not sleeping)
- Verify URL is correct

---

## 💡 Tips for Free Hosting Success

1. **Monitor Usage**: Check platform dashboards for usage limits
2. **Keep Logs Clean**: Don't log excessively
3. **Use Paper Trading**: Safer for testing
4. **Backup Data**: Free tiers may have data retention limits
5. **Stay Within Limits**: Monitor CPU/memory usage

---

## 🎯 Quick Start Summary

**Easiest FREE Option:**
1. Go to render.com → Sign up with GitHub
2. New Web Service → Connect `rkmatrix/aitrading`
3. Set build: `pip install -r requirements.txt`
4. Set start: `python -m dashboard.app`
5. Add environment variables
6. Deploy!
7. Set up UptimeRobot to keep awake

**Total Time: ~10 minutes**
**Total Cost: $0.00** ✅

---

**Your bot can now run 24/7 completely FREE! 🚀**
