/**
 * Main Dashboard JavaScript
 */

// Global state
const dashboardState = {
    botStatus: 'stopped',
    metrics: {},
    trades: [],
    positions: [],
    tickers: [],
    logs: [],
    currentTicker: null,
};

// Initialize dashboard
function initDashboard() {
    console.log('Dashboard initializing...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial data
    loadBotStatus();
    loadMetrics();
    loadTrades();
    loadTickers();
    loadPositions();
    
    // Setup auto-refresh
    setInterval(loadMetrics, 1000);
    setInterval(loadTrades, 2000);
    setInterval(loadPositions, 2000);
    setInterval(loadTickers, 5000); // Update ticker prices every 5 seconds
}

// Setup event listeners
function setupEventListeners() {
    // Bot control
    document.getElementById('bot-start-btn')?.addEventListener('click', startBot);
    document.getElementById('bot-stop-btn')?.addEventListener('click', stopBot);
    document.getElementById('kill-switch-toggle')?.addEventListener('click', toggleKillSwitch);
    
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tab = e.target.dataset.tab;
            if (tab) {
                switchTab(tab);
            }
        });
    });
    
    // Ticker search
    document.getElementById('ticker-search')?.addEventListener('input', handleTickerSearch);
    document.getElementById('add-ticker-btn')?.addEventListener('click', () => {
        openModal('ticker-search-modal');
    });
    
    // Settings
    document.getElementById('settings-btn')?.addEventListener('click', () => {
        openModal('settings-modal');
    });
    
    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
}

// Tab switching
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        }
    });
}

// Modal functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// Close modal on outside click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Bot control functions
async function startBot() {
    try {
        const response = await fetch('/api/bot/start', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            updateBotStatus('running');
        }
    } catch (error) {
        console.error('Failed to start bot:', error);
    }
}

async function stopBot() {
    try {
        const response = await fetch('/api/bot/stop', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            updateBotStatus('stopped');
        }
    } catch (error) {
        console.error('Failed to stop bot:', error);
    }
}

async function toggleKillSwitch() {
    try {
        const response = await fetch('/api/bot/kill-switch', { method: 'POST' });
        const data = await response.json();
        // Update UI based on response
    } catch (error) {
        console.error('Failed to toggle kill switch:', error);
    }
}

function updateBotStatus(status) {
    dashboardState.botStatus = status;
    const indicator = document.getElementById('bot-status-indicator');
    const statusText = document.getElementById('bot-status-text');
    const startBtn = document.getElementById('bot-start-btn');
    const stopBtn = document.getElementById('bot-stop-btn');
    
    if (indicator) {
        indicator.className = `status-indicator ${status}`;
    }
    
    if (statusText) {
        statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    if (startBtn && stopBtn) {
        if (status === 'running') {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
        } else {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        }
    }
}

// Data loading functions
async function loadBotStatus() {
    try {
        const response = await fetch('/api/bot/status');
        const data = await response.json();
        if (data.status) {
            updateBotStatus(data.status);
        }
    } catch (error) {
        console.error('Failed to load bot status:', error);
    }
}

async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics/current');
        const data = await response.json();
        if (data) {
            updateMetrics(data);
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

async function loadTrades() {
    try {
        const response = await fetch('/api/trades?limit=50');
        const data = await response.json();
        if (data.trades) {
            updateTradesTable(data.trades);
        }
    } catch (error) {
        console.error('Failed to load trades:', error);
    }
}

async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        const data = await response.json();
        if (data.positions) {
            updatePositionsTable(data.positions);
        }
    } catch (error) {
        console.error('Failed to load positions:', error);
    }
}

function updatePositionsTable(positions) {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody) return;
    
    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 24px;">No open positions</td></tr>';
        return;
    }
    
    tbody.innerHTML = positions.map(pos => `
        <tr>
            <td><strong>${pos.symbol}</strong></td>
            <td>${pos.qty}</td>
            <td>${formatCurrency(pos.avg_entry_price)}</td>
            <td>${formatCurrency(pos.current_price)}</td>
            <td>${formatCurrency(pos.market_value)}</td>
            <td style="color: ${pos.unrealized_pnl >= 0 ? 'var(--success-color)' : 'var(--error-color)'}">
                ${formatCurrency(pos.unrealized_pnl)}
            </td>
        </tr>
    `).join('');
}

async function loadTickers() {
    try {
        const response = await fetch('/api/tickers');
        const data = await response.json();
        if (data.tickers) {
            updateTickerList(data.tickers);
        }
    } catch (error) {
        console.error('Failed to load tickers:', error);
    }
}

// Update functions
function updateMetrics(metrics) {
    if (metrics.equity !== undefined) {
        document.getElementById('metric-equity').textContent = formatCurrency(metrics.equity);
    }
    if (metrics.buying_power !== undefined) {
        document.getElementById('metric-buying-power').textContent = formatCurrency(metrics.buying_power);
    }
    if (metrics.total_pnl !== undefined) {
        const pnlEl = document.getElementById('metric-total-pnl');
        pnlEl.textContent = formatCurrency(metrics.total_pnl);
        pnlEl.style.color = metrics.total_pnl >= 0 ? 'var(--success-color)' : 'var(--error-color)';
    }
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('trades-tbody');
    if (!tbody) return;
    
    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 24px;">No trades yet</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${formatTime(trade.timestamp)}</td>
            <td><strong>${trade.symbol}</strong></td>
            <td style="color: ${trade.side === 'BUY' ? 'var(--success-color)' : 'var(--error-color)'}">${trade.side}</td>
            <td>${trade.qty}</td>
            <td>${formatCurrency(trade.price)}</td>
            <td><span class="status-badge ${trade.status.toLowerCase()}">${trade.status}</span></td>
            <td>${trade.pnl ? formatCurrency(trade.pnl) : '-'}</td>
        </tr>
    `).join('');
}

function updateTickerList(tickers) {
    const list = document.getElementById('ticker-list');
    if (!list) return;
    
    list.innerHTML = tickers.map(ticker => `
        <li class="ticker-item" data-symbol="${ticker.symbol}" onclick="openTickerAnalysis('${ticker.symbol}')">
            <div>
                <div class="ticker-symbol">${ticker.symbol}</div>
                <div class="ticker-price" style="font-size: 12px; color: var(--text-secondary);">${ticker.price ? formatCurrency(ticker.price) : '-'}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div class="ticker-change ${ticker.change >= 0 ? 'positive' : 'negative'}">${ticker.change ? (ticker.change >= 0 ? '+' : '') + ticker.change.toFixed(2) + '%' : '-'}</div>
                <button class="btn-icon" style="width: 24px; height: 24px; font-size: 12px;" onclick="event.stopPropagation(); removeTicker('${ticker.symbol}')">×</button>
            </div>
        </li>
    `).join('');
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function toggleTheme() {
    const body = document.body;
    if (body.classList.contains('theme-dark')) {
        body.classList.remove('theme-dark');
        body.classList.add('theme-light');
    } else {
        body.classList.remove('theme-light');
        body.classList.add('theme-dark');
    }
}

function handleTickerSearch(e) {
    const query = e.target.value.trim();
    if (query.length >= 1) {
        searchTickers(query);
    } else {
        document.getElementById('search-results').classList.remove('active');
    }
}

async function searchTickers(query) {
    try {
        const response = await fetch(`/api/tickers/search?q=${query}`);
        const data = await response.json();
        // Display search results
    } catch (error) {
        console.error('Search failed:', error);
    }
}

function openTickerAnalysis(symbol) {
    dashboardState.currentTicker = symbol;
    document.getElementById('analysis-modal-title').textContent = `${symbol} Analysis`;
    openModal('ticker-analysis-modal');
    loadTickerAnalysis(symbol);
}

async function loadTickerAnalysis(symbol) {
    try {
        const response = await fetch(`/api/ticker/${symbol}/analysis`);
        const data = await response.json();
        // Update analysis modal with data
    } catch (error) {
        console.error('Failed to load ticker analysis:', error);
    }
}

async function removeTicker(symbol) {
    if (!confirm(`Remove ${symbol} from trading list?`)) return;
    
    try {
        const response = await fetch(`/api/tickers/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol })
        });
        const data = await response.json();
        if (data.success) {
            loadTickers();
        }
    } catch (error) {
        console.error('Failed to remove ticker:', error);
    }
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}
