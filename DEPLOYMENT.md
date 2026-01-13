# Deployment Guide - AITradingBot

This guide explains how to deploy the AITradingBot and Dashboard to free hosting platforms.

## Prerequisites

1. **Git Repository**: Your code should be pushed to GitHub/GitLab/Bitbucket
2. **API Keys**: Alpaca API credentials (paper or live)
3. **Account**: Account on your chosen hosting platform

## Free Hosting Options

### 1. Railway (Recommended)

**Pros:**
- Free tier with $5 credit/month
- Easy Docker deployment
- Automatic HTTPS
- Good for Python apps

**Steps:**

1. **Sign up**: https://railway.app
2. **Create new project** → "Deploy from GitHub repo"
3. **Select your repository**
4. **Configure environment variables:**
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   MODE=PAPER
   DASHBOARD_SECRET_KEY=generate-random-string
   PORT=5000
   ```
5. **Set start command**: `python -m dashboard.app`
6. **Deploy**

**Railway-specific files:**
- Uses `dashboard/Dockerfile` if present
- Or detects Python and installs from `requirements.txt`

### 2. Render

**Pros:**
- Free tier available
- Easy Flask deployment
- Automatic HTTPS
- Good documentation

**Steps:**

1. **Sign up**: https://render.com
2. **Create new Web Service**
3. **Connect GitHub repository**
4. **Configure:**
   - **Name**: aitradingbot-dashboard
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m dashboard.app`
5. **Add environment variables** (same as Railway)
6. **Deploy**

**Render-specific files:**
- Uses `dashboard/Procfile` if present
- Or detects from start command

### 3. Fly.io

**Pros:**
- Free tier with generous limits
- Docker-based
- Global edge deployment
- Good performance

**Steps:**

1. **Install Fly CLI**: https://fly.io/docs/getting-started/installing-flyctl/
2. **Sign up**: `fly auth signup`
3. **Create app**: `fly launch`
4. **Configure `fly.toml`:**
   ```toml
   app = "aitradingbot-dashboard"
   primary_region = "iad"
   
   [build]
     dockerfile = "dashboard/Dockerfile"
   
   [env]
     PORT = "5000"
     FLASK_ENV = "production"
   
   [[services]]
     internal_port = 5000
     protocol = "tcp"
   ```
5. **Set secrets**: `fly secrets set APCA_API_KEY_ID=your_key ...`
6. **Deploy**: `fly deploy`

### 4. PythonAnywhere

**Pros:**
- Free tier for Python apps
- Easy Flask deployment
- No Docker needed

**Steps:**

1. **Sign up**: https://www.pythonanywhere.com
2. **Create new Web App** → Flask
3. **Upload code** via Git or files
4. **Configure WSGI file** to point to `dashboard/app.py`
5. **Set environment variables** in Web App settings
6. **Reload**

### 5. Heroku (Limited Free Tier)

**Pros:**
- Well-documented
- Easy deployment
- Good add-ons

**Cons:**
- Requires credit card (even for free tier)
- Limited free dyno hours

**Steps:**

1. **Install Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli
2. **Login**: `heroku login`
3. **Create app**: `heroku create aitradingbot-dashboard`
4. **Set config vars**: `heroku config:set APCA_API_KEY_ID=your_key ...`
5. **Deploy**: `git push heroku main`

**Heroku-specific files:**
- Uses `dashboard/Procfile`
- Uses `dashboard/runtime.txt` for Python version

## Environment Variables

Set these in your hosting platform's environment variables section:

### Required:
```bash
APCA_API_KEY_ID=your_alpaca_api_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MODE=PAPER
```

### Recommended:
```bash
DASHBOARD_SECRET_KEY=generate-random-secret-key-here
FLASK_ENV=production
PORT=5000
```

### Optional:
```bash
DATABASE_URL=postgresql://...  # For production database
TELEGRAM_BOT_TOKEN=...         # For Telegram alerts
TELEGRAM_CHAT_ID=...           # For Telegram alerts
```

## Generating Secret Keys

**Python:**
```python
import secrets
print(secrets.token_hex(32))
```

**Online:**
- https://randomkeygen.com/
- Use a strong random string generator

## Deployment Checklist

- [ ] Code pushed to Git repository
- [ ] `.env.example` created (template only, no secrets)
- [ ] `.gitignore` excludes `.env` and sensitive files
- [ ] Environment variables set in hosting platform
- [ ] `DASHBOARD_SECRET_KEY` set to strong random string
- [ ] `MODE` set to `PAPER` for testing
- [ ] Database configured (if using PostgreSQL)
- [ ] Domain/URL configured (if needed)
- [ ] HTTPS enabled (usually automatic)
- [ ] Test deployment works
- [ ] Monitor logs for errors

## Post-Deployment

1. **Test the dashboard**: Visit your deployment URL
2. **Check logs**: Monitor for any errors
3. **Test bot start/stop**: Use dashboard controls
4. **Verify API connection**: Check if Alpaca API works
5. **Monitor resources**: Watch CPU/memory usage

## Troubleshooting

### Dashboard won't start
- Check environment variables are set correctly
- Verify `PORT` is set (usually auto-set by hosting)
- Check logs for Python errors
- Ensure `requirements.txt` installs successfully

### Bot won't connect to Alpaca
- Verify API keys are correct
- Check `APCA_API_BASE_URL` matches your mode (paper vs live)
- Ensure `MODE` is set correctly
- Check network/firewall settings

### Database errors
- For SQLite: Ensure `dashboard/data/` directory is writable
- For PostgreSQL: Verify `DATABASE_URL` is correct
- Check database migrations ran successfully

### Out of memory
- Free tiers have limited RAM
- Consider upgrading or optimizing code
- Reduce concurrent operations
- Use lighter dependencies

## Security Best Practices

1. **Never commit `.env` file** - Use `.env.example` template
2. **Use strong secret keys** - Generate random strings
3. **Enable HTTPS** - Most platforms do this automatically
4. **Limit API permissions** - Use read-only keys if possible
5. **Monitor access logs** - Watch for suspicious activity
6. **Rotate secrets regularly** - Change keys periodically
7. **Use environment variables** - Never hardcode secrets

## Cost Considerations

**Free Tier Limits:**
- Railway: $5 credit/month
- Render: 750 hours/month (sleeps after inactivity)
- Fly.io: 3 shared VMs, 160GB outbound data
- PythonAnywhere: 1 web app, limited CPU
- Heroku: 550-1000 dyno hours/month

**Upgrade when:**
- Need 24/7 uptime (free tiers may sleep)
- Need more resources (CPU/RAM)
- Need custom domains
- Need database backups
- Need more bandwidth

## Next Steps

1. Deploy dashboard first (test it works)
2. Test bot start/stop from dashboard
3. Monitor for a few days
4. Gradually enable more features
5. Consider upgrading if needed

## Support

- Check hosting platform documentation
- Review application logs
- Test locally first before deploying
- Use paper trading mode for testing
