"""
Box Trading Performance Validator
---------------------------------
Validates paper trading performance before allowing live trading.
Analyzes trade history and ensures all success criteria are met.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BoxTradingValidator")


class PerformanceValidator:
    """Validates box trading performance against requirements"""
    
    def __init__(self, config_path: str = "configs/box_trading.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.requirements = self.config.get("paper_trading_requirements", {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'box_trading' in config:
            config = config['box_trading']
        
        return config
    
    def load_trade_history(self, history_file: str = "data/box_trading_history.json") -> List[Dict[str, Any]]:
        """Load trade history from file"""
        history_path = Path(history_file)
        
        if not history_path.exists():
            logger.warning(f"Trade history file not found: {history_file}")
            return []
        
        try:
            with open(history_path, 'r') as f:
                trades = json.load(f)
            
            logger.info(f"Loaded {len(trades)} trades from {history_file}")
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            return []
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_hold_time": 0.0,
                "max_consecutive_losses": 0,
                "sharpe_ratio": 0.0,
                "duration_days": 0
            }
        
        # Separate wins and losses
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        
        # Calculate P&L
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_pnl = sum(t.get("pnl", 0) for t in wins)
        loss_pnl = abs(sum(t.get("pnl", 0) for t in losses))
        
        # Win rate
        win_rate = len(wins) / len(trades) if trades else 0.0
        
        # Profit factor
        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 0.0
        
        # Averages
        avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.get("pnl", 0) for t in losses]) if losses else 0.0
        
        # Largest
        largest_win = max([t.get("pnl", 0) for t in wins]) if wins else 0.0
        largest_loss = min([t.get("pnl", 0) for t in losses]) if losses else 0.0
        
        # Max drawdown
        cumulative_pnl = [0]
        for trade in trades:
            cumulative_pnl.append(cumulative_pnl[-1] + trade.get("pnl", 0))
        
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Max consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.get("pnl", 0) <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Average hold time
        hold_times = [t.get("duration_minutes", 0) for t in trades if "duration_minutes" in t]
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0
        
        # Sharpe ratio (simplified)
        if trades:
            returns = [t.get("pnl", 0) for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Duration
        if trades:
            try:
                first_date = datetime.fromisoformat(trades[0].get("entry_time", ""))
                last_date = datetime.fromisoformat(trades[-1].get("exit_time", ""))
                duration_days = (last_date - first_date).days
            except:
                duration_days = 0
        else:
            duration_days = 0
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "max_drawdown_percent": (max_drawdown / 10000) if max_drawdown > 0 else 0.0,  # Assuming $10k starting
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_hold_time": avg_hold_time,
            "max_consecutive_losses": max_consecutive,
            "sharpe_ratio": sharpe_ratio,
            "duration_days": duration_days
        }
    
    def validate_requirements(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate metrics against requirements
        Returns (passed, failures, warnings)
        """
        failures = []
        warnings = []
        
        # Minimum trades
        min_trades = self.requirements.get("min_trades", 50)
        if metrics["total_trades"] < min_trades:
            failures.append(f"Insufficient trades: {metrics['total_trades']} < {min_trades} required")
        
        # Win rate
        min_win_rate = self.requirements.get("min_win_rate", 0.55)
        if metrics["win_rate"] < min_win_rate:
            failures.append(f"Win rate too low: {metrics['win_rate']*100:.1f}% < {min_win_rate*100:.0f}% required")
        
        # Profit factor
        min_profit_factor = self.requirements.get("min_profit_factor", 1.4)
        if metrics["profit_factor"] < min_profit_factor:
            failures.append(f"Profit factor too low: {metrics['profit_factor']:.2f} < {min_profit_factor} required")
        
        # Max drawdown
        max_drawdown = self.requirements.get("max_drawdown_percent", 0.08)
        if metrics["max_drawdown_percent"] > max_drawdown:
            failures.append(f"Drawdown too high: {metrics['max_drawdown_percent']*100:.1f}% > {max_drawdown*100:.0f}% allowed")
        
        # Duration
        min_duration = self.requirements.get("min_duration_days", 28)
        if metrics["duration_days"] < min_duration:
            failures.append(f"Insufficient testing period: {metrics['duration_days']} days < {min_duration} days required")
        
        # Warnings (not failures, but should review)
        if metrics["max_consecutive_losses"] > 5:
            warnings.append(f"High consecutive losses: {metrics['max_consecutive_losses']} (review strategy)")
        
        if metrics["largest_loss"] < metrics["avg_loss"] * 3:
            warnings.append(f"Large loss outlier detected: ${metrics['largest_loss']:.2f} vs avg ${metrics['avg_loss']:.2f}")
        
        if metrics["sharpe_ratio"] < 1.0:
            warnings.append(f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f} (< 1.0)")
        
        if metrics["avg_hold_time"] > 150:
            warnings.append(f"Long average hold time: {metrics['avg_hold_time']:.0f} minutes (expected <120)")
        
        passed = len(failures) == 0
        
        return (passed, failures, warnings)
    
    def generate_report(self, metrics: Dict[str, Any], passed: bool, failures: List[str], warnings: List[str]) -> str:
        """Generate validation report"""
        report = []
        report.append("=" * 80)
        report.append("BOX TRADING PERFORMANCE VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Status
        if passed:
            report.append("✅ VALIDATION PASSED - Ready for Live Trading")
        else:
            report.append("❌ VALIDATION FAILED - Continue Paper Trading")
        report.append("")
        
        # Metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 80)
        report.append(f"Total Trades:          {metrics['total_trades']}")
        report.append(f"Win Rate:              {metrics['win_rate']*100:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
        report.append(f"Profit Factor:         {metrics['profit_factor']:.2f}")
        report.append(f"Total P&L:             ${metrics['total_pnl']:.2f}")
        report.append(f"Max Drawdown:          ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_percent']*100:.1f}%)")
        report.append(f"Avg Win:               ${metrics['avg_win']:.2f}")
        report.append(f"Avg Loss:              ${metrics['avg_loss']:.2f}")
        report.append(f"Largest Win:           ${metrics['largest_win']:.2f}")
        report.append(f"Largest Loss:          ${metrics['largest_loss']:.2f}")
        report.append(f"Avg Hold Time:         {metrics['avg_hold_time']:.0f} minutes")
        report.append(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        report.append(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
        report.append(f"Testing Duration:      {metrics['duration_days']} days")
        report.append("")
        
        # Requirements
        report.append("REQUIREMENTS CHECK:")
        report.append("-" * 80)
        
        if failures:
            report.append("❌ FAILURES:")
            for failure in failures:
                report.append(f"   - {failure}")
            report.append("")
        
        if warnings:
            report.append("⚠️  WARNINGS:")
            for warning in warnings:
                report.append(f"   - {warning}")
            report.append("")
        
        if not failures and not warnings:
            report.append("✅ All checks passed with no warnings!")
            report.append("")
        
        # Next steps
        report.append("NEXT STEPS:")
        report.append("-" * 80)
        
        if passed:
            report.append("1. Review all metrics carefully")
            report.append("2. Ensure you understand why wins won and losses lost")
            report.append("3. Change ENV to LIVE in .env file")
            report.append("4. Update config: current_phase: 'live_phase1'")
            report.append("5. Start with TINY position sizes ($100 max)")
            report.append("6. Monitor every single trade")
            report.append("7. Continue for 2 weeks before increasing size")
        else:
            report.append("1. Continue paper trading")
            report.append("2. Address all failures listed above")
            report.append("3. Analyze losing trades for patterns")
            report.append("4. Consider adjusting:")
            report.append("   - Zone thresholds")
            report.append("   - Confirmation requirements")
            report.append("   - Symbol selection")
            report.append("5. Re-run validation when requirements met")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_validation(self, history_file: str = "data/box_trading_history.json") -> bool:
        """
        Run full validation
        Returns True if passed, False otherwise
        """
        logger.info("Starting validation...")
        
        # Load trades
        trades = self.load_trade_history(history_file)
        
        if not trades:
            print("\n❌ No trade history found!")
            print(f"Expected file: {history_file}")
            print("\nPlease run the bot in paper trading mode first.")
            return False
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        # Validate
        passed, failures, warnings = self.validate_requirements(metrics)
        
        # Generate report
        report = self.generate_report(metrics, passed, failures, warnings)
        
        # Print report
        print("\n" + report)
        
        # Save report
        report_file = Path("data/box_trading_validation_report.txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
            f.write(f"\n\nGenerated: {datetime.now().isoformat()}\n")
        
        logger.info(f"Report saved to: {report_file}")
        
        return passed


def main():
    """Entry point"""
    print("=" * 80)
    print("BOX TRADING PERFORMANCE VALIDATOR")
    print("=" * 80)
    print()
    
    validator = PerformanceValidator()
    
    # Check for custom history file
    history_file = "data/box_trading_history.json"
    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    
    passed = validator.run_validation(history_file)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
