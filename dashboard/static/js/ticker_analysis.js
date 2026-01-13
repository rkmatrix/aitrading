/**
 * Ticker Analysis Modal JavaScript
 */

// Tab switching in analysis modal
document.querySelectorAll('[data-analysis-tab]').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const tab = e.target.dataset.analysisTab;
        
        // Update active tab button
        document.querySelectorAll('[data-analysis-tab]').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.analysis-tab-content').forEach(content => {
            content.classList.remove('active');
            if (content.id === `analysis-${tab}-tab`) {
                content.classList.add('active');
            }
        });
        
        // Load tab-specific data
        if (tab === 'chart' && dashboardState.currentTicker) {
            initTickerChart(dashboardState.currentTicker, '1m');
        } else if (tab === 'overview' && dashboardState.currentTicker) {
            loadOverviewData(dashboardState.currentTicker);
        }
    });
});

// Chart period buttons
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('chart-period-btn')) {
        document.querySelectorAll('.chart-period-btn').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        const period = e.target.dataset.period;
        if (dashboardState.currentTicker) {
            initTickerChart(dashboardState.currentTicker, period);
        }
    }
});

async function loadTickerAnalysis(symbol) {
    try {
        const response = await fetch(`/api/ticker/${symbol}/analysis`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Analysis error:', data.error);
            return;
        }
        
        // Update overview tab
        if (data.overview) {
            loadOverviewData(symbol, data.overview);
        }
        
        // Update performance metrics
        if (data.performance) {
            document.getElementById('analysis-win-rate').textContent = 
                `${(data.performance.win_rate * 100).toFixed(1)}%`;
            document.getElementById('analysis-avg-return').textContent = 
                `${(data.performance.avg_return * 100).toFixed(2)}%`;
            document.getElementById('analysis-total-trades').textContent = 
                data.performance.total_trades;
            document.getElementById('analysis-cumulative-pnl').textContent = 
                formatCurrency(data.performance.cumulative_pnl);
        }
        
        // Update risk metrics
        if (data.risk) {
            document.getElementById('analysis-position-size').textContent = 
                data.risk.position_size || '0';
            document.getElementById('analysis-exposure').textContent = 
                `${(data.risk.exposure_pct * 100).toFixed(2)}%`;
            document.getElementById('analysis-volatility').textContent = 
                `${(data.risk.volatility * 100).toFixed(2)}%`;
        }
        
        // Load trades for this ticker
        if (data.trades) {
            updateTickerTradesTable(data.trades);
        }
        
        // Load signals
        if (data.signals) {
            updateSignalsList(data.signals);
        }
        
        // Initialize chart with default period
        initTickerChart(symbol, '1m');
        
    } catch (error) {
        console.error('Failed to load ticker analysis:', error);
    }
}

function loadOverviewData(symbol, overviewData = null) {
    if (!overviewData) {
        // Fetch overview data if not provided
        fetch(`/api/market/ticker/${symbol}/info`)
            .then(r => r.json())
            .then(data => {
                if (data.symbol) {
                    displayOverviewData(data);
                }
            })
            .catch(err => console.error('Failed to load overview:', err));
    } else {
        displayOverviewData(overviewData);
    }
}

function displayOverviewData(data) {
    // Company name and exchange
    document.getElementById('overview-company-name').textContent = data.name || data.symbol;
    const exchangeInfo = [];
    if (data.exchange) exchangeInfo.push(data.exchange);
    if (data.sector) exchangeInfo.push(data.sector);
    if (data.industry) exchangeInfo.push(data.industry);
    document.getElementById('overview-exchange-sector').textContent = exchangeInfo.join(' • ') || '';
    
    // Price and change
    const price = data.current_price || 0;
    const change = data.change || 0;
    const changePercent = data.change_percent || 0;
    const changeColor = change >= 0 ? 'var(--success-color)' : 'var(--error-color)';
    const changeSign = change >= 0 ? '+' : '';
    
    document.getElementById('overview-price').textContent = formatCurrency(price);
    document.getElementById('overview-change').innerHTML = 
        `<span style="color: ${changeColor}">${changeSign}${formatCurrency(Math.abs(change))} (${changeSign}${changePercent.toFixed(2)}%)</span>`;
    
    // Market cap
    const marketCap = data.market_cap || 0;
    document.getElementById('overview-market-cap').textContent = formatLargeNumber(marketCap);
    
    // P/E Ratio
    const pe = data.pe_ratio || 0;
    document.getElementById('overview-pe').textContent = pe > 0 ? pe.toFixed(2) : 'N/A';
    
    // Volume
    const volume = data.volume || 0;
    document.getElementById('overview-volume').textContent = formatLargeNumber(volume);
    
    // 52-week high/low
    document.getElementById('overview-52high').textContent = formatCurrency(data.week52_high || 0);
    document.getElementById('overview-52low').textContent = formatCurrency(data.week52_low || 0);
    
    // Description
    const desc = data.description || 'No description available.';
    document.getElementById('overview-description').textContent = desc;
}

function formatLargeNumber(num) {
    if (!num || num === 0) return '0';
    if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
}

// Ensure formatCurrency and formatTime are available globally
if (typeof formatCurrency === 'undefined') {
    window.formatCurrency = function(value) {
        if (value === null || value === undefined || isNaN(value)) return '$0.00';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    };
}

if (typeof formatTime === 'undefined') {
    window.formatTime = function(timestamp) {
        if (!timestamp) return '-';
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    };
}

function updateTickerTradesTable(trades) {
    const tbody = document.getElementById('ticker-trades-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${formatTime(trade.timestamp)}</td>
            <td style="color: ${trade.side === 'BUY' ? 'var(--success-color)' : 'var(--error-color)'}">${trade.side}</td>
            <td>${trade.qty}</td>
            <td>${formatCurrency(trade.price)}</td>
            <td style="color: ${trade.pnl >= 0 ? 'var(--success-color)' : 'var(--error-color)'}">
                ${trade.pnl ? formatCurrency(trade.pnl) : '-'}
            </td>
        </tr>
    `).join('');
}

function updateSignalsList(signals) {
    const container = document.getElementById('signals-list');
    if (!container) return;
    
    if (!signals || signals.length === 0) {
        container.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-secondary);">No signals available</div>';
        return;
    }
    
    container.innerHTML = signals.map(signal => {
        const actionColor = signal.action === 'BUY' ? 'var(--success-color)' : 'var(--error-color)';
        return `
        <div class="signal-item" style="padding: 16px; border-bottom: 1px solid var(--border-color); transition: background-color 0.2s;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                <div>
                    <strong style="color: ${actionColor}; font-size: 14px;">${signal.action}</strong>
                    <span class="status-badge ${signal.status.toLowerCase()}" style="margin-left: 8px; font-size: 11px;">${signal.status}</span>
                </div>
                <span style="font-size: 12px; color: var(--text-secondary);">${formatTime(signal.timestamp)}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; font-size: 12px; color: var(--text-secondary);">
                <div>
                    <span style="font-weight: 500;">Confidence:</span> ${(signal.confidence * 100).toFixed(1)}%
                </div>
                <div>
                    <span style="font-weight: 500;">Quantity:</span> ${signal.qty}
                </div>
                <div>
                    <span style="font-weight: 500;">Price:</span> ${formatCurrency(signal.price)}
                </div>
                ${signal.filled_qty ? `
                    <div>
                        <span style="font-weight: 500;">Filled:</span> ${signal.filled_qty}
                    </div>
                ` : ''}
            </div>
        </div>
        `;
    }).join('');
}
