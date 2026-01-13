# Git Setup Guide - AITradingBot

This guide explains how to set up Git for the AITradingBot project and prepare it for deployment.

## Initial Git Setup

### 1. Initialize Git Repository (if not already done)

```bash
cd C:\Projects\trading\AITradeBot_core
git init
```

### 2. Verify .gitignore is in place

The `.gitignore` file should exclude:
- `.env` files (contains secrets)
- `data/` directory (trading data, logs, reports)
- `models/` directory (trained models)
- `__pycache__/` directories
- Database files
- Log files

Check that `.gitignore` exists and contains the exclusions.

### 3. Create .env from Template

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your actual API keys
# DO NOT commit .env to Git!
```

### 4. Check What Will Be Committed

```bash
# See what files Git will track
git status

# See what files are ignored
git status --ignored
```

**Expected output:**
- ✅ Should track: `ai/`, `runner/`, `dashboard/`, `configs/`, `tools/`, `tests/`
- ❌ Should ignore: `data/`, `models/`, `.env`, `__pycache__/`, `*.log`

## Files That SHOULD Be in Git

### Critical Code Files:
```
✅ ai/                    # All AI modules
✅ runner/                # Execution scripts
✅ tools/                 # Utility scripts
✅ tests/                 # Test files
✅ dashboard/             # Dashboard code (except dashboard/data/)
✅ configs/               # Configuration YAML files
✅ data/policies_src/     # Policy source templates
✅ *.md                   # Documentation files
✅ .gitignore             # Git ignore rules
✅ .env.example           # Environment template
✅ requirements.txt       # Python dependencies
✅ dashboard/Dockerfile   # Docker config
✅ dashboard/Procfile     # Heroku/Railway config
✅ dashboard/runtime.txt  # Python version
```

## Files That Should NOT Be in Git

### Data Files:
```
❌ data/*.csv            # Training data
❌ data/features/         # Feature files
❌ data/backtests/        # Backtest results
❌ data/logs/            # Log files
❌ data/runtime/         # Runtime state
❌ data/reports/         # Generated reports
❌ data/models/          # Model files
```

### Model Files:
```
❌ models/               # Entire directory
❌ *.pt, *.pth          # PyTorch models
❌ *.h5, *.keras        # Keras models
❌ *.joblib, *.pkl     # Scikit-learn models
❌ *.zip                # Compressed models
```

### Secrets and Config:
```
❌ .env                  # Environment variables (SECRETS!)
❌ .env.local
❌ *.env
```

### Runtime Files:
```
❌ dashboard/data/      # Dashboard database
❌ __pycache__/         # Python cache
❌ *.log                # Log files
❌ *.db                 # Database files
```

## Step-by-Step Git Setup

### Step 1: Verify .gitignore

```bash
# Check .gitignore exists
cat .gitignore

# Verify it excludes .env
grep -i "\.env" .gitignore
```

### Step 2: Create .env File (Don't Commit!)

```bash
# Copy template
cp .env.example .env

# Edit with your API keys
# Use your favorite editor: notepad .env, nano .env, etc.
```

### Step 3: Check Git Status

```bash
git status
```

**You should see:**
- ✅ Files to be committed: Code files, configs, docs
- ❌ Ignored files: `.env`, `data/`, `models/`, `__pycache__/`

### Step 4: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what was added
git status
```

**Important:** Make sure `.env` is NOT in the list!

### Step 5: Create Initial Commit

```bash
git commit -m "Initial commit: AITradingBot core code and dashboard"
```

### Step 6: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `AITradingBot`)
3. **DO NOT** initialize with README (you already have files)
4. Copy the repository URL

### Step 7: Connect and Push

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/AITradingBot.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Verifying What's in Git

### Check Tracked Files

```bash
# List all tracked files
git ls-files

# Count files
git ls-files | wc -l

# Check specific directories
git ls-files | grep "^data/"
git ls-files | grep "^models/"
git ls-files | grep "\.env"
```

**Expected:**
- ✅ `data/` should only show `data/policies_src/` (source templates)
- ✅ `models/` should be empty (no tracked files)
- ✅ `.env` should NOT appear

### Check Ignored Files

```bash
# See what's ignored
git status --ignored

# Check specific files
git check-ignore -v data/logs/
git check-ignore -v models/
git check-ignore -v .env
```

## Common Mistakes to Avoid

### ❌ DON'T Commit These:

1. **`.env` file** - Contains API keys and secrets
2. **`data/` directory** - Large data files, logs, reports
3. **`models/` directory** - Trained models are huge
4. **`__pycache__/`** - Python cache files
5. **`*.log` files** - Log files
6. **Database files** - `*.db` files

### ✅ DO Commit These:

1. **`.env.example`** - Template (no secrets)
2. **`requirements.txt`** - Dependencies list
3. **`configs/`** - Configuration YAML files
4. **`data/policies_src/`** - Policy templates
5. **All `.py` files** - Source code
6. **All `.md` files** - Documentation

## If You Accidentally Committed Secrets

### Remove from Git History:

```bash
# Remove .env from Git (but keep local file)
git rm --cached .env

# Commit the removal
git commit -m "Remove .env from Git"

# If already pushed, you need to force push (DANGEROUS!)
# git push --force

# IMPORTANT: Rotate your API keys immediately!
# The old keys are in Git history and should be considered compromised
```

## Repository Size Check

```bash
# Check repository size
git count-objects -vH

# If repository is too large, check for large files
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort --numeric-sort --key=2 --reverse | head -20
```

**If repository is >100MB:**
- Check for large files in `data/` or `models/`
- Ensure `.gitignore` is working
- Consider Git LFS for large files (if needed)

## Pre-Push Checklist

Before pushing to GitHub:

- [ ] `.env` file exists locally but is NOT in Git
- [ ] `.env.example` exists and is in Git
- [ ] `data/` directory is ignored (except `policies_src/`)
- [ ] `models/` directory is ignored
- [ ] `__pycache__/` directories are ignored
- [ ] `*.log` files are ignored
- [ ] `requirements.txt` is up to date
- [ ] All code files are tracked
- [ ] Documentation files are tracked
- [ ] No secrets in any committed files

## Testing Before Push

```bash
# Create a test branch
git checkout -b test-branch

# Make a test commit
echo "test" > test.txt
git add test.txt
git commit -m "Test commit"

# Verify what would be pushed
git ls-tree -r test-branch --name-only

# Delete test branch
git checkout main
git branch -D test-branch
rm test.txt
```

## Next Steps

1. ✅ Git repository initialized
2. ✅ `.gitignore` configured
3. ✅ `.env.example` created
4. ✅ Files committed locally
5. ✅ Pushed to GitHub
6. ✅ Ready for deployment

See `DEPLOYMENT.md` for deployment instructions.

## Security Reminder

**NEVER commit:**
- API keys
- Secret keys
- Passwords
- `.env` files
- Personal data

**ALWAYS:**
- Use `.env.example` as template
- Rotate keys if accidentally committed
- Review commits before pushing
- Use environment variables in hosting platform
