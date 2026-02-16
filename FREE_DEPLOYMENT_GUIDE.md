# Free Cloud Deployment Guide - Step by Step

This guide provides detailed step-by-step instructions to deploy your AITradingBot for **FREE** and run it **24/7 continuously**.

## 🎯 Best Free Options

1. **Railway** (Recommended) - $5 free credit/month, easiest setup
2. **Render** - Free tier available, good for Flask apps
3. **Fly.io** - Free tier, good Docker support
4. **PythonAnywhere** - Free tier, simple Python hosting

---

## 🚀 Option 1: Railway (RECOMMENDED - Easiest)

### Step 1: Create Railway Account
1. Go to **https://railway.app**
2. Click **"Start a New Project"** or **"Login"**
3. Sign up with GitHub (recommended) or email
4. Verify your email if required

### Step 2: Deploy from GitHub
1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway to access your GitHub if prompted
4. Find and select your repository: **`rkmatrix/aitrading`**
5. Click **"Deploy Now"**

### Step 3: Configure Environment Variables
1. In your Railway project, click on the deployed service
2. Go to **"Variables"** tab
3. Click **"New Variable"** and add each of these:

```
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
DASHBOARD_SECRET_KEY=generate-a-random-secret-key-here-min-32-chars
PORT=5000
FLASK_ENV=production
```

**To generate a secure DASHBOARD_SECRET_KEY:**
- Use: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Or use any random 32+ character string

### Step 4: Configure Start Command
1. In Railway project settings, go to **"Settings"** tab
2. Find **"Start Command"** section
3. Set it to: `python -m dashboard.app`
4. Save changes

### Step 5: Deploy
1. Railway will automatically detect your code and start building
2. Watch the build logs - it will install dependencies
3. Once deployed, Railway will provide a URL like: `https://your-app.up.railway.app`
4. Click the URL to access your dashboard

### Step 6: Keep It Running (Free Tier)
- Railway free tier gives $5 credit/month
- Your bot uses minimal resources, should run for free
- Monitor usage in Railway dashboard
- If you exceed free tier, Railway will pause (not delete) your app

**✅ Done!** Your bot is now running 24/7 at the Railway URL.

---

## 🌐 Option 2: Render (Good Free Alternative)

### Step 1: Create Render Account
1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with GitHub (recommended)

### Step 2: Create Web Service
1. In Render dashboard, click **"New +"**
2. Select **"Web Service"**
3. Click **"Connect GitHub"** and authorize Render
4. Find and select repository: **`rkmatrix/aitrading`**
5. Click **"Connect"**

### Step 3: Configure Service Settings
Fill in these settings:

- **Name**: `aitradingbot-dashboard` (or any name)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main`
- **Root Directory**: (leave blank)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m dashboard.app`

### Step 4: Add Environment Variables
Scroll down to **"Environment Variables"** section and click **"Add Environment Variable"**:

Add each variable:
```
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
DASHBOARD_SECRET_KEY=your-random-secret-key-here
PORT=5000
FLASK_ENV=production
```

### Step 5: Choose Free Plan
1. Scroll to **"Plan"** section
2. Select **"Free"** plan
3. Click **"Create Web Service"**

### Step 6: Deploy
1. Render will start building your app
2. Watch build logs - wait for "Your service is live"
3. Render provides URL like: `https://aitradingbot-dashboard.onrender.com`
4. **Note**: Free tier apps sleep after 15 minutes of inactivity
   - To keep it awake, use a service like **UptimeRobot** (free) to ping your URL every 5 minutes

**✅ Done!** Your bot is deployed on Render.

---

## 🪶 Option 3: Fly.io (Docker-Based)

### Step 1: Install Fly CLI
**Windows:**
1. Download from: https://fly.io/docs/hands-on/install-flyctl/
2. Or use PowerShell:
   ```powershell
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

**Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Sign Up
1. Run: `flyctl auth signup`
2. Follow prompts to create account
3. Verify email

### Step 3: Launch App
1. Navigate to your project directory:
   ```bash
   cd C:\Projects\trading\AITradeBot_core
   ```

2. Launch app:
   ```bash
   flyctl launch
   ```

3. Follow prompts:
   - **App name**: `aitradingbot` (or choose your own)
   - **Region**: Choose closest (e.g., `iad` for Virginia)
   - **PostgreSQL**: Answer **"No"** (we use SQLite)
   - **Deploy now**: Answer **"Yes"**

### Step 4: Set Secrets (Environment Variables)
```bash
flyctl secrets set APCA_API_KEY_ID=your_key
flyctl secrets set APCA_API_SECRET_KEY=your_secret
flyctl secrets set APCA_API_BASE_URL=https://paper-api.alpaca.markets
flyctl secrets set MODE=PAPER
flyctl secrets set DASHBOARD_SECRET_KEY=your-random-secret-key
flyctl secrets set FLASK_ENV=production
```

### Step 5: Deploy
```bash
flyctl deploy
```

### Step 6: Open App
```bash
flyctl open
```

**✅ Done!** Your bot is running on Fly.io.

---

## 🐍 Option 4: PythonAnywhere (Simple Python Hosting)

### Step 1: Create Account
1. Go to **https://www.pythonanywhere.com**
2. Click **"Create a Beginner account"** (free)
3. Sign up with email

### Step 2: Upload Code
1. Login to PythonAnywhere dashboard
2. Go to **"Files"** tab
3. Click **"Upload a file"** and upload your project as ZIP
   - Or use **"Bash"** tab and clone from GitHub:
     ```bash
     git clone https://github.com/rkmatrix/aitrading.git
     ```

### Step 3: Create Web App
1. Go to **"Web"** tab
2. Click **"Add a new web app"**
3. Choose **"Flask"**
4. Python version: **3.10** or **3.11**
5. Path: `/home/yourusername/aitrading/dashboard/app.py`
6. Click **"Next"** → **"Next"** → **"Finish"**

### Step 4: Configure WSGI File
1. Click **"WSGI configuration file"** link
2. Edit the file to point to your app:
   ```python
   import sys
   path = '/home/yourusername/aitrading'
   if path not in sys.path:
       sys.path.insert(0, path)
   
   from dashboard.app import app as application
   ```

### Step 5: Set Environment Variables
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

### Step 6: Install Dependencies
1. Go to **"Tasks"** tab
2. Click **"Bash"** to open terminal
3. Run:
   ```bash
   cd aitrading
   pip3.10 install --user -r requirements.txt
   ```

### Step 7: Reload Web App
1. Go back to **"Web"** tab
2. Click **"Reload"** button
3. Your app URL: `https://yourusername.pythonanywhere.com`

**✅ Done!** Your bot is running on PythonAnywhere.

---

## 🔄 Keep Free Apps Awake (Render/PythonAnywhere)

Free tiers on Render and PythonAnywhere sleep after inactivity. Keep them awake:

### Using UptimeRobot (Free)
1. Go to **https://uptimerobot.com**
2. Sign up for free account
3. Click **"Add New Monitor"**
4. Configure:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: AITradingBot
   - **URL**: Your app URL (e.g., `https://your-app.onrender.com`)
   - **Monitoring Interval**: 5 minutes
5. Click **"Create Monitor"**

This will ping your app every 5 minutes to keep it awake.

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] Your code is pushed to GitHub (`rkmatrix/aitrading`)
- [ ] You have Alpaca API credentials (paper trading)
- [ ] You've generated a secure `DASHBOARD_SECRET_KEY`
- [ ] You've tested the dashboard locally (`python -m dashboard.app`)

---

## 🎯 Recommended: Railway

**Why Railway?**
- ✅ Easiest setup (just connect GitHub)
- ✅ $5 free credit/month (usually enough)
- ✅ Automatic HTTPS
- ✅ No sleep (stays awake)
- ✅ Good documentation
- ✅ Easy environment variable management

**Quick Start:**
1. Sign up at railway.app
2. Deploy from GitHub → Select `rkmatrix/aitrading`
3. Add environment variables
4. Set start command: `python -m dashboard.app`
5. Done!

---

## 🆘 Troubleshooting

### Build Fails
- Check build logs in your hosting platform
- Ensure `requirements.txt` exists and is complete
- Verify Python version (3.10+)

### App Crashes
- Check runtime logs
- Verify all environment variables are set
- Ensure database is initialized (run `python dashboard/init_db.py` if needed)

### Can't Access Dashboard
- Check if app is running (not sleeping)
- Verify PORT environment variable
- Check firewall/security settings

### Bot Not Trading
- Verify API keys are correct
- Check MODE is set to PAPER (not LIVE)
- Review bot logs in dashboard

---

## 📞 Next Steps After Deployment

1. **Access Dashboard**: Open your deployment URL
2. **Monitor Bot**: Check dashboard for bot status, trades, metrics
3. **View Logs**: Use dashboard logs panel or hosting platform logs
4. **Set Up Alerts**: Configure Telegram alerts (optional)
5. **Keep It Running**: Monitor usage to stay within free tier limits

---

## 💡 Tips for Free Tier Success

1. **Monitor Usage**: Check your hosting platform's usage dashboard
2. **Optimize Resources**: Free tiers have limits - monitor CPU/memory
3. **Use Paper Trading**: Safer for testing, uses same resources
4. **Keep Logs Clean**: Don't log excessively to save storage
5. **Backup Important Data**: Free tiers may have data retention limits

---

**Your bot is now ready to run 24/7 in the cloud! 🚀**
