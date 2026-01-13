/**
 * Trades JavaScript
 */

// Trade filtering
document.getElementById('trades-filter')?.addEventListener('input', (e) => {
    const filter = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#trades-tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(filter) ? '' : 'none';
    });
});

// Export trades
document.getElementById('export-trades-btn')?.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/trades/export');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trades_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
    } catch (error) {
        console.error('Export failed:', error);
    }
});
