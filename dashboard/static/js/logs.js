/**
 * Logs JavaScript
 */

// Log level filtering
document.querySelectorAll('[data-log-level]').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const level = e.target.dataset.logLevel;
        
        // Update active button
        document.querySelectorAll('[data-log-level]').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        // Filter logs
        const logs = document.querySelectorAll('.log-entry');
        logs.forEach(log => {
            if (level === 'all' || log.classList.contains(level)) {
                log.style.display = '';
            } else {
                log.style.display = 'none';
            }
        });
    });
});

// Log search
document.getElementById('logs-search')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const logs = document.querySelectorAll('.log-entry');
    
    logs.forEach(log => {
        const text = log.textContent.toLowerCase();
        log.style.display = text.includes(query) ? '' : 'none';
    });
});

// Clear logs
document.getElementById('clear-logs-btn')?.addEventListener('click', () => {
    const logsPanel = document.getElementById('logs-panel');
    if (logsPanel && confirm('Clear all logs?')) {
        logsPanel.innerHTML = '';
    }
});
