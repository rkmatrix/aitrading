/**
 * Tickers JavaScript
 */

async function addTicker(symbol) {
    try {
        const response = await fetch('/api/tickers/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol })
        });
        const data = await response.json();
        if (data.success) {
            loadTickers();
            closeModal('ticker-search-modal');
        }
    } catch (error) {
        console.error('Failed to add ticker:', error);
    }
}

async function haltTicker(symbol) {
    try {
        const response = await fetch('/api/tickers/halt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol })
        });
        const data = await response.json();
        if (data.success) {
            loadTickers();
        }
    } catch (error) {
        console.error('Failed to halt ticker:', error);
    }
}

async function resumeTicker(symbol) {
    try {
        const response = await fetch('/api/tickers/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol })
        });
        const data = await response.json();
        if (data.success) {
            loadTickers();
        }
    } catch (error) {
        console.error('Failed to resume ticker:', error);
    }
}

// Ticker search in header
let searchTimeout;
document.getElementById('ticker-search')?.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    const query = e.target.value.toUpperCase().trim();
    
    if (query.length >= 1) {
        searchTimeout = setTimeout(() => {
            searchTickers(query);
        }, 300);
    } else {
        document.getElementById('search-results').classList.remove('active');
    }
});

let searchTimeout;
let selectedIndex = -1;
let searchResults = [];

async function searchTickers(query) {
    clearTimeout(searchTimeout);
    
    if (!query || query.length < 1) {
        document.getElementById('search-results').classList.remove('active');
        return;
    }
    
    searchTimeout = setTimeout(async () => {
        try {
            const response = await fetch(`/api/market/search?q=${encodeURIComponent(query)}&limit=20`);
            const data = await response.json();
            
            const resultsDiv = document.getElementById('search-results');
            searchResults = data.results || [];
            selectedIndex = -1;
            
            if (searchResults.length > 0) {
                resultsDiv.innerHTML = searchResults.map((ticker, idx) => {
                    const changeColor = ticker.change_percent >= 0 ? 'var(--success-color)' : 'var(--error-color)';
                    const changeSign = ticker.change_percent >= 0 ? '+' : '';
                    return `
                        <div class="search-result-item ${idx === selectedIndex ? 'selected' : ''}" 
                             onclick="selectTicker('${ticker.symbol}')"
                             onmouseover="highlightSearchResult(${idx})"
                             data-index="${idx}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 600; font-size: 14px;">${ticker.symbol}</div>
                                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                                        ${ticker.name || ticker.symbol}
                                    </div>
                                    ${ticker.exchange ? `<div style="font-size: 11px; color: var(--text-secondary);">${ticker.exchange}</div>` : ''}
                                </div>
                                <div style="text-align: right;">
                                    ${ticker.current_price ? `<div style="font-weight: 600;">${formatCurrency(ticker.current_price)}</div>` : ''}
                                    ${ticker.change_percent !== undefined ? `
                                        <div style="font-size: 12px; color: ${changeColor};">
                                            ${changeSign}${ticker.change_percent.toFixed(2)}%
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
                resultsDiv.classList.add('active');
            } else {
                resultsDiv.innerHTML = '<div style="padding: 12px; color: var(--text-secondary); text-align: center;">No results found</div>';
                resultsDiv.classList.add('active');
            }
        } catch (error) {
            console.error('Search failed:', error);
            document.getElementById('search-results').innerHTML = '<div style="padding: 12px; color: var(--error-color);">Search error. Please try again.</div>';
        }
    }, 300); // Debounce 300ms
}

function highlightSearchResult(index) {
    selectedIndex = index;
    document.querySelectorAll('.search-result-item').forEach((item, idx) => {
        item.classList.toggle('selected', idx === index);
    });
}

// Keyboard navigation for search results
document.addEventListener('keydown', (e) => {
    const resultsDiv = document.getElementById('search-results');
    if (!resultsDiv.classList.contains('active') || searchResults.length === 0) return;
    
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, searchResults.length - 1);
        highlightSearchResult(selectedIndex);
        document.querySelectorAll('.search-result-item')[selectedIndex]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, -1);
        if (selectedIndex >= 0) {
            highlightSearchResult(selectedIndex);
            document.querySelectorAll('.search-result-item')[selectedIndex]?.scrollIntoView({ block: 'nearest' });
        }
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
        e.preventDefault();
        selectTicker(searchResults[selectedIndex].symbol);
    } else if (e.key === 'Escape') {
        resultsDiv.classList.remove('active');
        selectedIndex = -1;
    }
});

function selectTicker(symbol) {
    document.getElementById('ticker-search').value = symbol;
    document.getElementById('search-results').classList.remove('active');
    openTickerAnalysis(symbol);
}
