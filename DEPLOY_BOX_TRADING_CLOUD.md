# Box Trading Bot - Cloud Deployment Guide

## 🚀 Deploy Your Bot to Run 24/7 in the Cloud

This guide will help you deploy the Box Trading Bot to run continuously without your local machine.

---

## 📋 **Recommended Cloud Platforms**

### **Option 1: Render.com (RECOMMENDED)** ⭐

**Best for your situation because:**
- ✅ You already use it (main bot deployed there)
- ✅ Free tier available ($0/month)
- ✅ Easy deployment from GitHub
- ✅ Built-in logging dashboard
- ✅ Automatic restarts on crashes
- ✅ Environment variable management
- ✅ Can run multiple services (main bot + box bot)

**Limitations:**
- Free tier sleeps after 15 min inactivity
- Need paid plan ($7/month) for 24/7 worker

---

### **Option 2: PythonAnywhere**

**Good because:**
- ✅ Specialized for Python
- ✅ Free tier available
- ✅ Always-on scheduled tasks
- ✅ SSH access for debugging
- ✅ $5/month for always-on

**Limitations:**
- Free tier has CPU/bandwidth limits
- More manual setup

---

### **Option 3: AWS EC2 (Advanced)**

**Best for serious trading:**
- ✅ Full control
- ✅ High reliability (99.99% uptime)
- ✅ Scalable
- ✅ Professional grade

**Limitations:**
- More complex setup
- Costs ~$7-15/month
- Requires AWS knowledge

---

## 🎯 **OPTION 1: Render.com Deployment (EASIEST)**

### **Architecture:**

You can run **both bots simultaneously** on Render:

```
Your Render.com Account
├── Service 1: Main Trading Bot (existing)
│   └── Command: python runner/phase26_realtime_live.py
│   └── Symbols: Your current symbols
│
└── Service 2: Box Trading Bot (NEW)
    └── Command: python runner/box_trading_runner.py
    └── Symbols: SPY, QQQ, AAPL, MSFT
```

**Cost:** $7/month per service = $14/month total for both bots running 24/7

---

## 📝 **Step-by-Step: Render Deployment**

### **Step 1: Create Procfile.box**

Already created! But verify it exists in your repo.

### **Step 2: Update requirements.txt** (if needed)

Ensure all dependencies are listed:
```txt
alpaca-trade-api>=3.0.0
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
requests>=2.31.0
python-telegram-bot>=20.0
pytz
python-dotenv
pandas-market-calendars
```

### **Step 3: Push to GitHub** (if any changes)

```bash
git add Procfile.box requirements.txt
git commit -m "Add Render deployment config for box trading bot"
git push origin main
```

### **Step 4: Create New Service on Render**

1. **Go to:** https://dashboard.render.com
2. **Click:** "New +" → "Background Worker"
3. **Select Repository:** rkmatrix/aitrading
4. **Configure:**

```
Service Name:     box-trading-bot
Region:           Oregon (US West) - or closest to you
Branch:           main
Root Directory:   (leave empty)
Runtime:          Python 3
Build Command:    pip install -r requirements.txt
Start Command:    python runner/box_trading_runner.py
Instance Type:    Starter ($7/month) - for 24/7 operation
```

### **Step 5: Add Environment Variables**

In Render dashboard, go to "Environment" tab and add:

```
ENV                  = PAPER_TRADING
APCA_API_KEY_ID      = <your_alpaca_paper_key>
APCA_API_SECRET_KEY  = <your_alpaca_paper_secret>
APCA_API_BASE_URL    = https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN   = <your_telegram_bot_token>
TELEGRAM_CHAT_ID     = <your_telegram_chat_id>
TELEGRAM_ENABLED     = true
```

⚠️ **NEVER commit .env to Git** - always use Render's environment variables!

### **Step 6: Deploy**

1. Click **"Create Background Worker"**
2. Render will start deploying
3. Watch the deployment log
4. Wait for "Deploy live" message (2-3 minutes)

### **Step 7: Verify It's Running**

**Check Render Logs:**
- Should see: "Box Trading Bot Starting"
- Should see: "Market closed - idling" (if after hours)
- Should see: "Alpaca client initialized"

**Check Telegram:**
- Should receive: "🚀 BOX TRADING BOT STARTED" alert
- Should show symbols and validation date

**Success!** Your bot is now running 24/7 in the cloud! 🎉

---

## 📊 **Monitoring Your Cloud Bot**

### **Primary: Telegram Alerts**

You'll receive **all important notifications:**
- ✅ Startup/shutdown
- ✅ Every trade entry/exit
- ✅ Circuit breaker triggers
- ✅ Daily summaries (4:05 PM ET)
- ✅ 4-week validation reminder

**No need to check logs unless troubleshooting!**

### **Secondary: Render Dashboard**

**View Logs:**
1. Go to: https://dashboard.render.com
2. Click: "box-trading-bot" service
3. Click: "Logs" tab
4. See: Real-time streaming logs

**Features:**
- Search logs
- Download logs
- See last 7 days
- Real-time updates

**Metrics Tab:**
- CPU usage
- Memory usage
- Uptime
- Restart history

### **Health Checks:**

Render automatically monitors if your service crashes:
- ✅ Auto-restarts on failure
- ✅ Email alerts on crash (optional)
- ✅ Uptime tracking

---

## 🔄 **Auto-Deploy on Git Push**

**Huge advantage of Render:**

Every time you push to GitHub:
```bash
git push origin main
```

Render automatically:
1. ✅ Detects new commit
2. ✅ Pulls latest code
3. ✅ Rebuilds bot
4. ✅ Restarts with new version
5. ✅ Sends you notification

**No manual deployment needed!**

---

## 🛠️ **Managing Your Cloud Bot**

### **Restart Bot:**
- Render Dashboard → "Manual Deploy" → "Deploy latest commit"
- Or push a commit to trigger auto-deploy

### **Stop Bot:**
- Render Dashboard → Service Settings → "Suspend Service"
- Bot stops gracefully (closes positions first)

### **Update Configuration:**
- Edit `configs/box_trading.yaml` locally
- Commit and push
- Render auto-deploys
- Bot restarts with new config

### **Change to LIVE Trading:**
1. Go to Render Dashboard
2. Environment tab
3. Change `ENV` to `LIVE`
4. Change `APCA_API_BASE_URL` to `https://api.alpaca.markets`
5. Use LIVE API keys
6. Save (bot auto-restarts)

⚠️ **Only do this after validation passes!**

---

## 💡 **Best Practices for Cloud Deployment**

### **Security:**
- ✅ Never commit API keys to Git
- ✅ Always use environment variables
- ✅ Use paper trading first
- ✅ Monitor closely for first week

### **Reliability:**
- ✅ Enable Render's auto-restart
- ✅ Set up email alerts for crashes
- ✅ Monitor Telegram daily
- ✅ Check logs weekly

### **Cost Optimization:**
- ✅ Start with free tier (testing)
- ✅ Upgrade to $7/month when validated
- ✅ One service can run multiple strategies

### **Monitoring:**
- ✅ Telegram (primary)
- ✅ Render dashboard (secondary)
- ✅ Daily review routine
- ✅ Weekly performance check

---

## 🚨 **Troubleshooting Cloud Deployment**

### **Bot Not Starting:**

**Check Render Logs for:**
- "ModuleNotFoundError" → Add to requirements.txt
- "Missing API keys" → Check environment variables
- "Import errors" → Verify all dependencies

**Fix:**
1. Update requirements.txt
2. Push to GitHub
3. Render auto-redeploys

### **Bot Crashing:**

**Check Logs:**
- Look for error messages
- Check stack trace
- Identify root cause

**Common Issues:**
- Missing dependency
- API rate limits
- Network issues

**Solution:**
- Fix code
- Commit and push
- Auto-redeploys

### **No Telegram Alerts:**

**Check:**
- Environment variables set correctly
- Bot actually running (check logs)
- Telegram bot token valid
- Chat ID correct

---

## 📅 **Deployment Timeline**

### **Today (Setup):**
- 10 minutes: Create Render service
- 5 minutes: Configure environment
- 2 minutes: Deploy and verify
- ✅ Bot running in cloud!

### **Day 1-28 (Paper Trading):**
- Bot trades automatically
- You receive Telegram alerts
- Check logs occasionally
- Monitor performance

### **Day 28 (Validation):**
- Receive validation reminder
- Run validation tool locally
- Decide: continue paper or go live

### **Day 29+ (Optional Live):**
- Change ENV to LIVE on Render
- Start with tiny positions
- Monitor intensely

---

## 💰 **Total Cost Estimate**

### **Render.com (Recommended):**
```
Setup:           $0 (free)
First month:     $7 (or $0 if using free tier)
Ongoing:         $7/month per bot
Both bots:       $14/month
Annual:          $168/year for 24/7 operation
```

### **Compare to Running Locally:**
```
Your PC electricity: ~$10-20/month
Wear and tear:       Unknown
Reliability:         Depends on your PC/internet
Convenience:         Must keep PC on 24/7
```

**Cloud is actually CHEAPER and more reliable!**

---

## 🎯 **What I'll Do Next**

Let me create the deployment files for you:

1. ✅ Verify/create `Procfile.box`
2. ✅ Check `requirements.txt` is complete
3. ✅ Create `render.yaml` (optional, for automatic setup)
4. ✅ Push everything to Git
5. ✅ Give you exact Render dashboard settings

**Ready to proceed with Render.com deployment?**

Say "yes" and I'll create all the files and push them to your repo!
