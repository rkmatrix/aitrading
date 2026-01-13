# Files to Commit to Git - Quick Reference

## ✅ MUST Commit (Critical Code)

### Core Bot Code
- `ai/` - All Python modules (entire directory)
- `runner/` - Execution scripts
- `tools/` - Utility scripts  
- `tests/` - Test files
- Root Python files: `check_bot_status.py`, `monitor_bot.py`, etc.

### Dashboard Code
- `dashboard/` - Entire directory EXCEPT:
  - `dashboard/data/` (database files)
  - `dashboard/__pycache__/` (Python cache)

### Configuration
- `configs/` - All YAML configuration files
- `data/policies_src/` - Policy source templates (YAML/JSON)

### Documentation
- All `.md` files in root and `dashboard/`

### Deployment Files
- `dashboard/Dockerfile`
- `dashboard/docker-compose.yml`
- `dashboard/Procfile`
- `dashboard/runtime.txt`
- `dashboard/requirements.txt`
- `requirements.txt` (root)
- `dashboard/run.py`
- `dashboard/__main__.py`
- `dashboard/start_dashboard.bat`
- `dashboard/start_dashboard.sh`

### Git Configuration
- `.gitignore`
- `.gitattributes`
- `.env.example` (template, no secrets)

## ❌ DO NOT Commit (Data/Sample Files)

### Secrets
- `.env` - Contains API keys (NEVER commit!)
- `.env.local`
- Any file with `.env` extension

### Data Files
- `data/*.csv` - Training data
- `data/features/` - Feature files
- `data/backtests/` - Backtest results
- `data/logs/` - Log files
- `data/runtime/` - Runtime state
- `data/reports/` - Generated reports
- `data/pnl/` - P&L data
- `data/trades/` - Trade logs
- `data/orders/` - Order logs
- `data/policies/` - Generated policies (NOT `policies_src/`)

### Model Files
- `models/` - Entire directory
- `*.pt`, `*.pth` - PyTorch models
- `*.h5`, `*.keras` - Keras models
- `*.joblib`, `*.pkl` - Scikit-learn models
- `*.zip` - Compressed models

### Runtime Files
- `dashboard/data/` - Dashboard database
- `__pycache__/` - Python cache
- `*.log` - Log files
- `*.db` - Database files

### Old Files
- `phase26_realtime_live_PHASE_E_IDLE_NO_SYNTHETIC_FINAL.py` - Old version

## Quick Check Commands

### See what will be committed:
```bash
git status
```

### Verify .env is ignored:
```bash
git check-ignore .env
# Should output: .env
```

### Verify data/ is ignored (except policies_src):
```bash
git check-ignore data/logs/
# Should output: data/logs/

git check-ignore data/policies_src/
# Should output nothing (not ignored, will be tracked)
```

### Count files to be committed:
```bash
git ls-files | wc -l
```

## Pre-Commit Checklist

- [ ] `.env` file exists locally but is NOT in Git
- [ ] `.env.example` exists and IS in Git
- [ ] `data/policies_src/` will be tracked
- [ ] `data/` (except `policies_src/`) will be ignored
- [ ] `models/` will be ignored
- [ ] `dashboard/data/` will be ignored
- [ ] No secrets in any files
- [ ] All code files are tracked
- [ ] Documentation files are tracked

## Estimated Repository Size

**Will be committed:** ~6-12 MB (code, configs, docs)
**Will be excluded:** ~100-700 MB (data, models, logs)

This keeps your Git repository clean and fast! 🚀
