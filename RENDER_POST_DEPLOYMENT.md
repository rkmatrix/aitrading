# Render Post-Deployment Guide

## 🎉 Deployment Status: Building

Your `aitrading-dashboard` service is currently building on Render!

## ⏱️ What Happens Next

### Build Process (5-10 minutes)
1. Render clones your repository
2. Installs dependencies from `requirements.txt`
3. Builds your application
4. Starts the web service

### Monitor Build Progress
- Watch the **Logs** tab in Render dashboard
- You'll see build output in real-time
- Look for "Build successful" message

---

## ✅ Step 1: Add Environment Variables (CRITICAL)

**After build completes**, go to your service → **Settings** → **Environment** and add:

### Required Variables:

```bash
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
FLASK_SECRET_KEY=your_generated_secret_key_here
```

### Generate FLASK_SECRET_KEY:

Run this command locally:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output and paste it as `FLASK_SECRET_KEY` value.

### After Adding Variables:
- Click **"Save Changes"**
- Go to **Manual Deploy** → **"Deploy latest commit"**
- This redeploys with your environment variables

---

## ✅ Step 2: Verify Deployment

1. **Check Service Status**: Should show "Live" (green)
2. **Visit Your URL**: `https://aitrading-dashboard.onrender.com`
3. **Test Dashboard**: Should load your trading dashboard
4. **Check Logs**: Look for any errors in the Logs tab

---

## ✅ Step 3: Keep Service Awake (Free Tier)

**Important**: Render's free tier puts services to sleep after 15 minutes of inactivity.

### Option 1: UptimeRobot (Recommended - FREE)

1. **Sign up**: https://uptimerobot.com (free account)
2. **Add Monitor**:
   - Monitor Type: **HTTP(s)**
   - Friendly Name: `AITrading Dashboard`
   - URL: `https://aitrading-dashboard.onrender.com`
   - Monitoring Interval: **5 minutes**
3. **Save**: UptimeRobot will ping your service every 5 minutes
4. **Result**: Service stays awake 24/7! 🎉

### Option 2: Manual Ping
- Visit your dashboard URL every 10-15 minutes
- Or use a browser extension that auto-refreshes

---

## 🔍 Troubleshooting

### Build Failed?
1. Check **Logs** tab for error messages
2. Common issues:
   - Missing `requirements.txt` in root
   - Python version mismatch
   - Missing dependencies
   - Import errors

### Service Won't Start?
1. Check **Logs** for startup errors
2. Verify environment variables are set correctly
3. Check `Start Command`: Should be `python -m dashboard.app`
4. Verify `Root Directory`: Should be `dashboard`

### 502 Bad Gateway?
- Service might be sleeping (free tier)
- Wait 30 seconds and refresh
- Set up UptimeRobot to keep it awake

### Environment Variables Not Working?
- Make sure you clicked **"Save Changes"**
- Redeploy after adding variables: **Manual Deploy** → **Deploy latest commit**
- Check variable names match exactly (case-sensitive)

---

## 📊 Monitoring Your Service

### Render Dashboard Features:
- **Logs**: Real-time application logs
- **Metrics**: CPU, Memory, Network usage
- **Events**: Deployment history
- **Settings**: Environment variables, scaling, etc.

### Access Your Dashboard:
- **Render URL**: `https://aitrading-dashboard.onrender.com`
- **Local Development**: Still works at `http://localhost:5000`

---

## 🚀 Next Steps

1. ✅ **Wait for build to complete** (5-10 min)
2. ✅ **Add environment variables** (critical!)
3. ✅ **Redeploy** after adding variables
4. ✅ **Set up UptimeRobot** to keep service awake
5. ✅ **Test dashboard** at your Render URL
6. ✅ **Monitor logs** for any issues

---

## 📝 Quick Reference

### Your Service Info:
- **Service Name**: `aitrading-dashboard`
- **URL**: `https://aitrading-dashboard.onrender.com` (check Render dashboard for exact URL)
- **Repository**: `rkmatrix/aitrading`
- **Root Directory**: `dashboard`
- **Start Command**: `python -m dashboard.app`
- **Instance**: Free tier

### Important URLs:
- **Render Dashboard**: https://dashboard.render.com
- **UptimeRobot**: https://uptimerobot.com
- **Your Service**: Check Render dashboard for exact URL

---

## 🎯 Success Checklist

- [ ] Build completed successfully
- [ ] Environment variables added
- [ ] Service redeployed with variables
- [ ] Dashboard accessible at Render URL
- [ ] UptimeRobot monitor set up (optional but recommended)
- [ ] Logs show no errors
- [ ] Dashboard displays correctly

---

**Your bot is deploying! 🚀**

Monitor the build progress and follow the steps above once it completes.
