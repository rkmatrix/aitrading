# RENDER.COM DEPLOYMENT - QUICK GUIDE

## 🚀 Deploy Box Trading Bot to Render (10 Minutes)

### **Prerequisites:**
- ✅ GitHub account connected to Render
- ✅ Render.com account (free to create)
- ✅ Alpaca API keys ready
- ✅ Telegram bot token ready

---

## 📝 **Step-by-Step Instructions:**

### **1. Go to Render Dashboard**
👉 https://dashboard.render.com/

### **2. Create New Background Worker**
- Click: **"New +"** → **"Background Worker"**

### **3. Connect Repository**
- Select: **rkmatrix/aitrading** (your GitHub repo)
- Branch: **main**
- Render may auto-detect `render.yaml` configuration

### **4. Manual Configuration (if needed)**

```
Name:             box-trading-bot
Region:           Oregon (US West)
Branch:           main
Root Directory:   (leave blank)

Build Command:    pip install -r requirements.txt
Start Command:    python runner/box_trading_runner.py

Instance Type:    Starter ($7/month)
                  or Free (sleeps after 15min - only for testing)
```

### **5. Set Environment Variables**

Click "Environment" tab and add these **ONE BY ONE**:

```
Variable Name              | Value
---------------------------|----------------------------------------
ENV                        | PAPER_TRADING
APCA_API_KEY_ID            | <paste your Alpaca paper trading key>
APCA_API_SECRET_KEY        | <paste your Alpaca paper trading secret>
APCA_API_BASE_URL          | https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN         | <paste your Telegram bot token>
TELEGRAM_CHAT_ID           | <paste your Telegram chat ID>
TELEGRAM_ENABLED           | true
```

**Where to find these values:**
- Copy from your local `.env` file
- Alpaca keys: https://app.alpaca.markets/paper/dashboard
- Telegram token: From @BotFather chat
- Telegram chat ID: From @userinfobot

### **6. Deploy**
- Click: **"Create Background Worker"**
- Wait 2-3 minutes for deployment
- Status will change to "Live"

### **7. Verify Deployment**

**Check Logs:**
- Click "Logs" tab in Render
- Should see: "Box Trading Bot Starting"
- Should see: "✅ Broker initialized"

**Check Telegram:**
- Should receive: "🚀 BOX TRADING BOT STARTED"
- Shows symbols and 4-week validation date

**SUCCESS!** Bot is now running 24/7 in the cloud! 🎉

---

## 📊 **Monitoring Your Cloud Bot**

### **Daily Monitoring (5 minutes):**

1. **Check Telegram:**
   - Daily summary (4:05 PM ET)
   - Trade alerts throughout day
   - Circuit breaker notifications

2. **Verify Bot is Running:**
   - Render Dashboard → Check status shows "Live"
   - Check last log entry is recent

3. **Quick Stats Review:**
   - Win rate trending toward 55%+?
   - Any unusual losses?
   - Circuit breakers triggered?

### **Weekly Review (15 minutes):**

1. **Render Logs:**
   - Search for "ERROR" (any errors?)
   - Check "Position closed" entries
   - Review win/loss pattern

2. **Performance:**
   - Total trades for week
   - Weekly P&L
   - Best/worst symbols

3. **Adjustments:**
   - Blacklist underperforming symbols?
   - Adjust zone thresholds?
   - Changes needed in config?

---

## 🔄 **Making Changes to Cloud Bot**

### **Update Configuration:**

```bash
# Edit config locally
notepad configs\box_trading.yaml

# Commit and push
git add configs/box_trading.yaml
git commit -m "Adjust box trading parameters"
git push origin main
```

Render automatically:
- ✅ Detects push
- ✅ Rebuilds bot
- ✅ Restarts with new config
- ✅ No downtime

### **Update Code:**

```bash
# Make changes to strategy
notepad ai\strategies\box_trading_strategy.py

# Commit and push
git add ai/strategies/box_trading_strategy.py
git commit -m "Improve signal generation"
git push origin main
```

Auto-deploys to Render!

---

## 🛑 **Emergency Controls**

### **Stop Bot Immediately:**
1. Render Dashboard → box-trading-bot
2. Click "Suspend Service"
3. Bot stops (closes positions first)

### **Restart Bot:**
1. Render Dashboard → box-trading-bot
2. Click "Resume Service"
3. Bot restarts in seconds

### **View Errors:**
1. Click "Logs" tab
2. Search for "ERROR" or "WARNING"
3. Download logs if needed

---

## 💰 **Pricing Breakdown**

### **Free Tier (Testing Only):**
- ✅ Perfect for initial testing
- ✅ $0/month
- ❌ Sleeps after 15 min inactivity
- ❌ Not suitable for live trading

### **Starter Tier (Production):**
- ✅ $7/month
- ✅ Always running 24/7
- ✅ 512MB RAM (enough for bot)
- ✅ Auto-restart on crash
- ✅ Email alerts
- ⭐ **RECOMMENDED**

### **Professional Tier:**
- $25/month
- 2GB RAM
- Priority support
- Only needed if bot becomes complex

---

## 📞 **Support & Troubleshooting**

### **Bot Not Starting:**

**Check Render Logs for:**
```
ERROR: Missing environment variable
ERROR: Module not found
ERROR: API key invalid
```

**Solutions:**
1. Verify all environment variables set
2. Check requirements.txt has all dependencies
3. Validate API keys in Alpaca dashboard

### **Bot Crashing:**

**Render auto-restarts, but check:**
1. Logs tab → Error messages
2. Fix code → Push to GitHub
3. Auto-redeploys

### **No Trades Executing:**

**Check:**
1. Market hours (9:30 AM - 4:00 PM ET)
2. Is market open today? (not weekend/holiday)
3. Check logs: "skipped" messages show why
4. Trending markets = fewer signals (normal)

---

## ✅ **Deployment Checklist**

Before deploying, verify:

- [ ] requirements.txt includes all dependencies
- [ ] Procfile.box created
- [ ] render.yaml created
- [ ] All files pushed to GitHub
- [ ] Alpaca API keys ready (PAPER trading)
- [ ] Telegram bot token ready
- [ ] Render account created

---

## 🎯 **After Deployment**

1. **First Hour:**
   - Monitor Render logs continuously
   - Verify startup successful
   - Check Telegram alert received
   - Confirm bot detecting market hours correctly

2. **First Day:**
   - Check 2-3 times
   - Verify trades executing (if market conditions right)
   - Check daily summary at 4:05 PM ET

3. **First Week:**
   - Daily check of Telegram summaries
   - Weekly review of Render logs
   - Monitor for any errors

4. **Week 4:**
   - Receive validation reminder
   - Run validation tool
   - Decide next phase

---

## 🚀 **Ready to Deploy?**

All deployment files are created and ready to push!

Your bot will run **24/7 in the cloud** with:
- ✅ Automatic market hours handling
- ✅ Weekend idling
- ✅ Telegram monitoring
- ✅ Auto-restart on crashes
- ✅ Logs dashboard
- ✅ 4-week validation reminder

**Cost: $7/month for complete peace of mind!**
