# 🔧 Render Root Directory Fix

## Problem

Render is looking for `run.py` in the wrong location:
```
python: can't open file '/opt/render/project/src/run.py': [Errno 2] No such file or directory
```

This suggests the **Root Directory** setting might be incorrect.

## ✅ Solution Options

### Option 1: Change Root Directory to Empty (Recommended)

1. **Go to Render Settings**
   - Navigate to: https://dashboard.render.com/web/srv-d3hav0qli9vc73e06cug/settings
   - Scroll to "Build & Deploy" section

2. **Find Root Directory**
   - Look for "Root Directory" field
   - **Current**: `dashboard/`
   - **Change to**: (empty - leave blank)

3. **Update Start Command**
   - Find "Start Command" field
   - **Change to**: `python -m dashboard`
   - Click "Save Changes"

4. **Manual Deploy**
   - Click "Manual Deploy" → "Deploy latest commit"

### Option 2: Keep Root Directory as `dashboard/` and Fix Path

If Root Directory is `dashboard/`, the start command should be:
- **Start Command**: `python run.py`

But if it's still failing, try:
- **Start Command**: `cd dashboard && python run.py`

## Why Option 1 Works Better

Using `python -m dashboard`:
- ✅ Works from project root (Root Directory empty)
- ✅ Uses `dashboard/__main__.py` which handles paths correctly
- ✅ More reliable and standard approach
- ✅ Matches how the app is designed to run

## Updated Files

- ✅ `dashboard/Procfile` - Changed to `python -m dashboard`
- ✅ `dashboard/__main__.py` - Already handles path setup correctly

## After Fix

Once you update Root Directory and Start Command:
1. Render will redeploy automatically
2. Build should succeed ✅
3. Start command should work ✅
4. Service will be live at: http://aitradepro-api.onrender.com

---

**Update Root Directory to empty and Start Command to `python -m dashboard` in Render Settings!** 🚀
