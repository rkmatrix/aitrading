/**
 * WebSocket Client for Real-time Updates
 */

let socket = null;

function initWebSocket() {
    // Connect to Socket.IO server
    socket = io();
    
    socket.on('connect', () => {
        console.log('WebSocket connected');
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
    });
    
    // Listen for bot status updates
    socket.on('bot.status.update', (data) => {
        updateBotStatus(data.status);
    });
    
    // Listen for trade updates
    socket.on('trade.executed', (data) => {
        addTradeToTable(data.trade);
    });
    
    // Listen for metric updates
    socket.on('metric.update', (data) => {
        updateMetrics(data.metrics);
    });
    
    // Listen for log entries
    socket.on('log.entry', (data) => {
        addLogEntry(data.log);
    });
    
    // Listen for ticker status updates
    socket.on('ticker.status.update', (data) => {
        updateTickerStatus(data.symbol, data.status);
    });
    
    // Listen for errors
    socket.on('error.occurred', (data) => {
        console.error('Error from server:', data);
    });
}

function addTradeToTable(trade) {
    const tbody = document.getElementById('trades-tbody');
    if (!tbody) return;
    
    // Remove "no trades" message if present
    if (tbody.querySelector('td[colspan]')) {
        tbody.innerHTML = '';
    }
    
    // Add new trade at top
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${formatTime(trade.timestamp)}</td>
        <td><strong>${trade.symbol}</strong></td>
        <td style="color: ${trade.side === 'BUY' ? 'var(--success-color)' : 'var(--error-color)'}">${trade.side}</td>
        <td>${trade.qty}</td>
        <td>${formatCurrency(trade.price)}</td>
        <td><span class="status-badge ${trade.status.toLowerCase()}">${trade.status}</span></td>
        <td>${trade.pnl ? formatCurrency(trade.pnl) : '-'}</td>
    `;
    tbody.insertBefore(row, tbody.firstChild);
    
    // Highlight new row
    row.style.backgroundColor = 'rgba(41, 98, 255, 0.1)';
    setTimeout(() => {
        row.style.backgroundColor = '';
    }, 2000);
}

function addLogEntry(log) {
    const logsPanel = document.getElementById('logs-panel');
    if (!logsPanel) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${log.level.toLowerCase()}`;
    entry.innerHTML = `
        <span class="log-timestamp">${formatTime(log.timestamp)}</span>
        <span>[${log.level}]</span>
        <span>${log.message}</span>
    `;
    
    logsPanel.appendChild(entry);
    
    // Auto-scroll if enabled
    const autoScroll = document.getElementById('auto-scroll-logs');
    if (autoScroll && autoScroll.checked) {
        logsPanel.scrollTop = logsPanel.scrollHeight;
    }
    
    // Limit log entries (keep last 1000)
    while (logsPanel.children.length > 1000) {
        logsPanel.removeChild(logsPanel.firstChild);
    }
}

function updateTickerStatus(symbol, status) {
    const tickerItem = document.querySelector(`.ticker-item[data-symbol="${symbol}"]`);
    if (tickerItem) {
        // Update ticker status indicator
    }
}
