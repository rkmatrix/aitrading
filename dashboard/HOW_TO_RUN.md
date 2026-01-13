# How to Run the Dashboard

## ⚠️ Important: Run from the Correct Directory

The dashboard must be run from the **project root**, not from inside the dashboard directory.

## ✅ Correct Way (Project Root)

```bash
# Navigate to project root first
cd C:\Projects\trading\AITradeBot_core

# Then run
python -m dashboard.app
```

## ✅ Alternative: Use run.py (From Dashboard Directory)

If you're already in the dashboard directory:

```bash
cd C:\Projects\trading\AITradeBot_core\dashboard
python run.py
```

## ✅ Using Start Scripts

**Windows (from project root):**
```bash
dashboard\start_dashboard.bat
```

**Linux/Mac (from project root):**
```bash
bash dashboard/start_dashboard.sh
```

## ❌ Common Mistakes

### Wrong: Running from dashboard directory
```bash
cd C:\Projects\trading\AITradeBot_core\dashboard
python -m dashboard.app  # ❌ This will fail!
```

**Error:** `No module named dashboard.app`

**Why?** When you're inside the dashboard directory, Python doesn't see "dashboard" as a module - it sees the current directory as the root.

### Right: Running from project root
```bash
cd C:\Projects\trading\AITradeBot_core
python -m dashboard.app  # ✅ This works!
```

## Quick Reference

| Location | Command |
|----------|---------|
| Project Root | `python -m dashboard.app` |
| Dashboard Dir | `python run.py` |
| Windows Script | `dashboard\start_dashboard.bat` |
| Linux/Mac Script | `bash dashboard/start_dashboard.sh` |

## After Starting

Once running, open your browser to:
- **http://localhost:5000**
- **http://127.0.0.1:5000**

The dashboard will show:
- ✅ Ticker search in header
- ✅ Ticker list in left sidebar
- ✅ Calendar and widgets in right sidebar
- ✅ Main content area with metrics and trades
