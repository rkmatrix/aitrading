/**
 * Right Sidebar Widgets JavaScript
 */

async function loadMarketStatus() {
    try {
        const response = await fetch('/api/market/ticker/SPY/info'); // Use SPY as market proxy
        const data = await response.json();
        
        // Get market clock status
        const clockResponse = await fetch('/api/bot/status');
        const clockData = await clockResponse.json();
        
        const statusEl = document.getElementById('market-status-text');
        const timeEl = document.getElementById('market-status-time');
        
        if (statusEl && timeEl) {
            // This is simplified - you'd want to use actual market clock
            const now = new Date();
            const hour = now.getHours();
            const isWeekend = now.getDay() === 0 || now.getDay() === 6;
            
            if (isWeekend) {
                statusEl.textContent = 'Market Closed (Weekend)';
                statusEl.style.color = 'var(--text-secondary)';
            } else if (hour >= 9 && hour < 16) {
                statusEl.textContent = 'Market Open';
                statusEl.style.color = 'var(--success-color)';
            } else {
                statusEl.textContent = 'Market Closed';
                statusEl.style.color = 'var(--text-secondary)';
            }
            
            timeEl.textContent = now.toLocaleTimeString();
        }
    } catch (error) {
        console.error('Failed to load market status:', error);
    }
}

async function loadQuickStats() {
    try {
        const metricsResponse = await fetch('/api/metrics/current');
        const metrics = await metricsResponse.json();
        
        const positionsResponse = await fetch('/api/positions');
        const positions = await positionsResponse.json();
        
        // Update quick stats
        const dailyPnlEl = document.getElementById('quick-stat-daily-pnl');
        const positionsEl = document.getElementById('quick-stat-positions');
        const winRateEl = document.getElementById('quick-stat-win-rate');
        
        if (dailyPnlEl) {
            const dailyPnl = metrics.daily_pnl || 0;
            dailyPnlEl.textContent = formatCurrency(dailyPnl);
            dailyPnlEl.style.color = dailyPnl >= 0 ? 'var(--success-color)' : 'var(--error-color)';
        }
        
        if (positionsEl) {
            positionsEl.textContent = positions.positions ? positions.positions.length : 0;
        }
        
        // Calculate win rate from recent trades
        const tradesResponse = await fetch('/api/trades?limit=50');
        const trades = await tradesResponse.json();
        
        if (winRateEl && trades.trades) {
            const filledTrades = trades.trades.filter(t => t.status === 'FILLED');
            const winningTrades = filledTrades.filter(t => {
                // Would need PnL in trade data
                return true; // Placeholder
            });
            const winRate = filledTrades.length > 0 ? (winningTrades.length / filledTrades.length * 100) : 0;
            winRateEl.textContent = `${winRate.toFixed(1)}%`;
        }
    } catch (error) {
        console.error('Failed to load quick stats:', error);
    }
}

async function loadRecentActivity() {
    try {
        const tradesResponse = await fetch('/api/trades?limit=5');
        const trades = await tradesResponse.json();
        
        const activityEl = document.getElementById('recent-activity-widget');
        if (!activityEl) return;
        
        if (trades.trades && trades.trades.length > 0) {
            activityEl.innerHTML = trades.trades.map(trade => `
                <div style="padding: 8px; border-bottom: 1px solid var(--border-color); font-size: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>${trade.symbol}</strong> ${trade.side}</span>
                        <span style="color: var(--text-secondary);">${formatTime(trade.timestamp)}</span>
                    </div>
                    <div style="color: var(--text-secondary); margin-top: 4px;">
                        ${trade.qty} @ ${formatCurrency(trade.price)}
                    </div>
                </div>
            `).join('');
        } else {
            activityEl.innerHTML = '<div style="padding: 12px; color: var(--text-secondary); font-size: 12px;">No recent activity</div>';
        }
    } catch (error) {
        console.error('Failed to load recent activity:', error);
    }
}

// Load widgets on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        loadMarketStatus();
        loadQuickStats();
        loadRecentActivity();
        
        // Refresh every 30 seconds
        setInterval(() => {
            loadMarketStatus();
            loadQuickStats();
            loadRecentActivity();
        }, 30000);
    });
} else {
    loadMarketStatus();
    loadQuickStats();
    loadRecentActivity();
    
    setInterval(() => {
        loadMarketStatus();
        loadQuickStats();
        loadRecentActivity();
    }, 30000);
}
