# 🚀 Oracle Cloud Deployment - Complete Step-by-Step Guide
## Deploy Your Box Trading Bot 100% FREE Forever

---

## 📋 **What You'll Need (5 minutes prep)**

Before starting, have these ready:
- ✅ Email address (for Oracle account)
- ✅ Phone number (for verification)
- ✅ Your Alpaca API keys (from .env file)
- ✅ Your Telegram bot token and chat ID (from .env file)
- ⏰ Time needed: 30-45 minutes total

---

## 🎯 **Phase 1: Create Oracle Cloud Account (10 minutes)**

### **Step 1.1: Sign Up**

1. **Open browser** and go to: https://www.oracle.com/cloud/free/
2. **Click:** "Start for free" button
3. **Fill in details:**
   - Country: Select your country
   - Name: Your full name
   - Email: Your email address
   - Click "Verify my email"

4. **Check email** and click verification link

5. **Complete registration:**
   - Password: Create strong password
   - Company Name: (can use your name)
   - Cloud Account Name: Choose unique name (e.g., "yourname-trading")
   - Home Region: **IMPORTANT!** Choose closest region:
     - US: `US East (Ashburn)` or `US West (Phoenix)`
     - Europe: `Germany Central (Frankfurt)` or `UK South (London)`
     - Asia: `Japan East (Tokyo)` or `South Korea Central (Seoul)`
   
6. **Verify phone number** (you'll receive SMS code)

7. **Accept terms** and click "Start my free trial"

**⚠️ IMPORTANT:** Select "Always Free" resources only - NO credit card needed!

### **Step 1.2: Access Console**

After registration:
1. You'll see Oracle Cloud Console
2. Click "Get Started" or go directly to dashboard
3. You're ready to create your VM!

---

## 🖥️ **Phase 2: Create Virtual Machine (15 minutes)**

### **Step 2.1: Navigate to Compute**

1. In Oracle Cloud Console, click **☰** (hamburger menu) top-left
2. Go to: **Compute** → **Instances**
3. Click: **"Create Instance"** (big blue button)

### **Step 2.2: Configure Instance**

**Basic Configuration:**
```
Name: box-trading-bot
```

**Placement:**
- Availability Domain: (Leave default - AD-1)

**Image and Shape:**
1. **Image:** Click "Change Image"
   - Select: **Ubuntu** (Canonical Ubuntu 22.04)
   - Click: "Select Image"

2. **Shape:** Click "Change Shape"
   - **Shape Series:** Ampere (ARM-based) ⭐ RECOMMENDED
   - **Shape Name:** VM.Standard.A1.Flex
   - **Number of OCPUs:** 2 (or up to 4 if available)
   - **Amount of Memory (GB):** 12 (or up to 24 if available)
   - Click: "Select Shape"

**Why ARM/Ampere?** It's FREE and gives you 4 cores + 24GB RAM total (amazing!)

**Alternative (if ARM not available):**
   - Shape Series: AMD
   - Shape Name: VM.Standard.E2.1.Micro
   - This gives: 1 core + 1GB RAM (less powerful but still free)

### **Step 2.3: Networking**

**Primary VNIC:**
- Leave all defaults
- ✅ Assign a public IPv4 address (should be checked)

### **Step 2.4: SSH Keys** ⚠️ CRITICAL!

**Add SSH Keys:**
1. Select: **"Generate a key pair for me"** (recommended)
2. Click: **"Save Private Key"** 
   - Save to: `C:\Users\YourName\.ssh\oracle-key.key`
   - **DON'T LOSE THIS FILE!** You need it to access your VM
3. Click: **"Save Public Key"** (optional, but recommended)

**Alternative:** If you have existing SSH keys, you can paste public key

### **Step 2.5: Boot Volume**

- Size (GB): 50 (default is fine)
- Leave other settings default

### **Step 2.6: Create!**

1. Click: **"Create"** button (bottom of page)
2. **Wait 2-3 minutes** - Instance will show "Provisioning" then "Running"
3. Once "Running" - note down the **Public IP Address** (you'll need this!)

Example: `132.145.xxx.xxx`

---

## 🔒 **Phase 3: Configure Firewall (5 minutes)**

### **Step 3.1: Add Security Rule**

1. On Instance details page, scroll to **Primary VNIC**
2. Click on the **Subnet** link (e.g., "subnet-xxx")
3. Click on **Security Lists** (left sidebar)
4. Click on **Default Security List** 
5. Click: **"Add Ingress Rules"**

**Add Rule:**
```
Source Type:        CIDR
Source CIDR:        0.0.0.0/0
IP Protocol:        TCP
Destination Port:   22
Description:        SSH Access
```

6. Click: **"Add Ingress Rules"**

**✅ Done!** SSH port is now open.

---

## 💻 **Phase 4: Connect to VM (5 minutes)**

### **Step 4.1: Open PowerShell (Windows)**

1. Press `Win + X`
2. Select "Windows PowerShell" or "Terminal"

### **Step 4.2: Connect via SSH**

**Fix key permissions first:**
```powershell
# Navigate to where you saved the key
cd C:\Users\YourName\.ssh

# If using PowerShell 7+, fix permissions:
icacls oracle-key.key /inheritance:r
icacls oracle-key.key /grant:r "$($env:USERNAME):(R)"
```

**Connect to VM:**
```powershell
ssh -i oracle-key.key ubuntu@YOUR_PUBLIC_IP
```

Replace `YOUR_PUBLIC_IP` with IP from Step 2.6 (e.g., `132.145.xxx.xxx`)

**First time:** Type `yes` when asked about fingerprint

**✅ Success!** You should see Ubuntu welcome message and prompt: `ubuntu@box-trading-bot:~$`

---

## 🔧 **Phase 5: Install Dependencies (5 minutes)**

Now you're inside the VM. Run these commands:

### **Step 5.1: Update System**

```bash
sudo apt update && sudo apt upgrade -y
```

Wait 2-3 minutes for updates to complete.

### **Step 5.2: Install Python and Git**

```bash
sudo apt install python3-pip git -y
```

### **Step 5.3: Verify Installation**

```bash
python3 --version
pip3 --version
git --version
```

Should show:
- Python 3.10+ 
- pip 22+
- git 2.34+

**✅ Done!** System is ready.

---

## 📦 **Phase 6: Deploy Bot (10 minutes)**

### **Step 6.1: Clone Repository**

```bash
# Clone your bot repository
git clone https://github.com/rkmatrix/aitrading.git

# Enter directory
cd aitrading

# Verify files
ls -la
```

Should see: `runner/`, `ai/`, `configs/`, `requirements.txt`, etc.

### **Step 6.2: Install Dependencies**

```bash
pip3 install -r requirements.txt
```

**Wait 5-10 minutes** - this installs all Python packages (pandas, alpaca-trade-api, etc.)

**If you see errors about pandas:** Run this first:
```bash
sudo apt install python3-dev python3-setuptools -y
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### **Step 6.3: Create Environment File**

```bash
nano .env
```

**Paste your configuration** (replace with YOUR actual values):

```env
############################################################
# 🔹 CORE SETTINGS
############################################################
ENV=PAPER_TRADING
DRY_RUN=false
LOG_LEVEL=INFO

############################################################
# 🔹 ALPACA PAPER TRADING
############################################################
APCA_API_KEY_ID=YOUR_ALPACA_KEY_HERE
APCA_API_SECRET_KEY=YOUR_ALPACA_SECRET_HERE
APCA_API_BASE_URL=https://paper-api.alpaca.markets

############################################################
# 🔹 TELEGRAM ALERTS
############################################################
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID_HERE

############################################################
# 🔹 DATA SOURCES (Optional - but needed for enhanced features)
############################################################
DATA_SOURCE=yahoo
STREAMING=true
LOOKBACK_DAYS=60
```

**To save:**
1. Press `Ctrl + X`
2. Press `Y` (yes to save)
3. Press `Enter` (confirm filename)

### **Step 6.4: Test Run (Quick Check)**

```bash
python3 runner/box_trading_runner.py
```

**You should see:**
- "Box Trading Bot Starting"
- "✅ Broker initialized: PAPER_TRADING mode"
- "Market closed" (if outside market hours) OR signals being checked

**Press `Ctrl + C` to stop** (after verifying it works)

**✅ If you see those messages, bot works!**

---

## 🚀 **Phase 7: Configure Auto-Start Service (10 minutes)**

Now let's make the bot run 24/7 automatically, even after reboots!

### **Step 7.1: Create Systemd Service**

```bash
sudo nano /etc/systemd/system/boxtrading.service
```

**Paste this configuration:**

```ini
[Unit]
Description=Box Trading Bot - 24/7 Automated Trading
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/aitrading
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 /home/ubuntu/aitrading/runner/box_trading_runner.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Save:** `Ctrl+X`, `Y`, `Enter`

### **Step 7.2: Enable and Start Service**

```bash
# Reload systemd (load new service)
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable boxtrading.service

# Start service NOW
sudo systemctl start boxtrading.service

# Check status
sudo systemctl status boxtrading.service
```

**✅ Success looks like:**
```
● boxtrading.service - Box Trading Bot - 24/7 Automated Trading
   Loaded: loaded
   Active: active (running)
   ...
```

**If you see "active (running)" - SUCCESS!** 🎉

### **Step 7.3: View Logs**

```bash
# View real-time logs
sudo journalctl -u boxtrading.service -f

# View last 50 lines
sudo journalctl -u boxtrading.service -n 50

# View logs from today
sudo journalctl -u boxtrading.service --since today
```

**Press `Ctrl+C` to stop viewing logs**

---

## 📱 **Phase 8: Verify Telegram Alerts (2 minutes)**

Within 1-2 minutes, you should receive a Telegram message:

```
🚀 BOX TRADING BOT STARTED

Config: configs/box_trading.yaml
Symbols: SPY, QQQ, AAPL, MSFT
Max Positions: 2

📅 Validation Reminder: March 12, 2026
(Run validation after 4 weeks of paper trading)
```

**✅ If you received this - your bot is LIVE and running 24/7!**

---

## 🎛️ **Managing Your Bot**

### **View Logs in Real-Time:**
```bash
ssh -i oracle-key.key ubuntu@YOUR_PUBLIC_IP
sudo journalctl -u boxtrading.service -f
```

### **Check Bot Status:**
```bash
sudo systemctl status boxtrading.service
```

### **Restart Bot:**
```bash
sudo systemctl restart boxtrading.service
```

### **Stop Bot:**
```bash
sudo systemctl stop boxtrading.service
```

### **Start Bot:**
```bash
sudo systemctl start boxtrading.service
```

### **Update Bot (After Pushing Changes to GitHub):**
```bash
cd /home/ubuntu/aitrading
git pull origin main
pip3 install -r requirements.txt  # If dependencies changed
sudo systemctl restart boxtrading.service
```

### **View Recent Errors:**
```bash
sudo journalctl -u boxtrading.service -p err -n 20
```

### **Check Disk Space:**
```bash
df -h
```

### **Check Memory Usage:**
```bash
free -h
```

### **Check CPU Usage:**
```bash
top
```
(Press `q` to quit)

---

## 🔍 **Monitoring Your Bot**

### **Primary: Telegram Monitoring** ✅

You'll receive:
- 🚀 Startup notifications
- 🎯 Trade entry alerts (with complete details)
- ✅ Trade exit alerts (with P&L)
- ⚠️ Circuit breaker alerts
- 📊 Daily summaries (4:05 PM ET)
- 📅 4-week validation reminder

**This is your main monitoring tool!**

### **Secondary: SSH + Logs**

When you need details:
```bash
# Connect
ssh -i oracle-key.key ubuntu@YOUR_PUBLIC_IP

# Check status
sudo systemctl status boxtrading.service

# View logs
sudo journalctl -u boxtrading.service -f
```

### **Checking if Bot is Running:**

**Method 1: From Telegram**
- If bot is running, you'll get daily summaries
- If bot crashes, summaries stop

**Method 2: From SSH**
```bash
sudo systemctl is-active boxtrading.service
```
Shows: `active` or `inactive`

---

## 🚨 **Troubleshooting**

### **Bot Not Starting:**

```bash
# Check status
sudo systemctl status boxtrading.service

# View detailed logs
sudo journalctl -u boxtrading.service -n 100

# Common issues:
# 1. Missing dependencies
sudo apt install python3-dev python3-setuptools -y
pip3 install -r requirements.txt

# 2. Wrong path
ls -la /home/ubuntu/aitrading/runner/box_trading_runner.py

# 3. Permission issues
sudo chown -R ubuntu:ubuntu /home/ubuntu/aitrading
```

### **No Telegram Alerts:**

```bash
# Check if TELEGRAM_ENABLED=true in .env
cat .env | grep TELEGRAM

# Test manually
cd /home/ubuntu/aitrading
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('TELEGRAM_ENABLED'))"
```

### **Module Not Found Errors:**

```bash
# Reinstall dependencies
cd /home/ubuntu/aitrading
pip3 install -r requirements.txt --force-reinstall

# If pandas issues:
sudo apt install python3-dev python3-setuptools build-essential -y
pip3 install pandas --no-binary pandas
```

### **Bot Keeps Restarting:**

```bash
# View last crash
sudo journalctl -u boxtrading.service -n 200 | grep -i error

# Check for API issues
# Usually: Invalid API keys or network issues
```

---

## 🎉 **Success Checklist**

After deployment, verify:

- ✅ Oracle Cloud VM is "Running"
- ✅ SSH connection works
- ✅ Bot service is "active (running)"
- ✅ Telegram startup message received
- ✅ Bot logs show "Box Trading Bot Starting"
- ✅ No errors in logs (`sudo journalctl -u boxtrading.service -n 50`)
- ✅ Service enabled for auto-start (`sudo systemctl is-enabled boxtrading.service` shows "enabled")

**If all checks pass: Your bot is running 24/7 FREE forever!** 🚀

---

## 💰 **Cost Breakdown**

**Oracle Cloud Always Free:**
- VM Instance: $0
- Storage (50GB): $0
- Network (10TB): $0
- Public IP: $0

**Total Monthly Cost: $0**
**Total Annual Cost: $0**

**Compare to Render Starter:** Save $84/year!

---

## 📞 **Getting Help**

**If you encounter issues:**

1. **Check logs first:**
   ```bash
   sudo journalctl -u boxtrading.service -n 100
   ```

2. **Check service status:**
   ```bash
   sudo systemctl status boxtrading.service
   ```

3. **Verify .env file:**
   ```bash
   cat /home/ubuntu/aitrading/.env
   ```

4. **Test Python imports:**
   ```bash
   cd /home/ubuntu/aitrading
   python3 -c "import alpaca_trade_api; import pandas; print('OK')"
   ```

---

## 🎓 **Additional Resources**

**Oracle Cloud Documentation:**
- Console: https://cloud.oracle.com/
- Free Tier FAQ: https://www.oracle.com/cloud/free/faq.html

**Your Bot Documentation:**
- `README_BOX_TRADING.md` - Complete strategy guide
- `QUICKSTART_BOX_TRADING.md` - Quick start guide
- `configs/box_trading.yaml` - Configuration reference

---

## 🚀 **You're All Set!**

Your Box Trading Bot is now:
- ✅ Running 24/7 on Oracle Cloud
- ✅ Completely FREE forever
- ✅ Auto-restarts on crashes
- ✅ Auto-starts on VM reboot
- ✅ Monitored via Telegram
- ✅ Trading during market hours (9:30 AM - 4:00 PM ET)
- ✅ Idling on weekends and holidays

**Next steps:**
1. Monitor Telegram for daily summaries
2. After 4 weeks, run validation tool
3. Decide whether to continue paper trading or go live

**Good luck with your automated trading!** 📈🎉
