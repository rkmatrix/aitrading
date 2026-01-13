/**
 * Ticker Search Modal JavaScript
 */

let modalSearchTimeout;
let modalSearchResults = [];
let selectedModalIndex = -1;

function initTickerSearchModal() {
    const searchInput = document.getElementById('ticker-search-input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', (e) => {
        clearTimeout(modalSearchTimeout);
        const query = e.target.value.trim();
        
        if (query.length < 1) {
            document.getElementById('ticker-search-results-modal').innerHTML = 
                '<div style="padding: 24px; text-align: center; color: var(--text-secondary);">Start typing to search for tickers...</div>';
            return;
        }
        
        modalSearchTimeout = setTimeout(() => {
            searchTickersModal(query);
        }, 300);
    });
    
    // Keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
        const resultsDiv = document.getElementById('ticker-search-results-modal');
        if (!resultsDiv || modalSearchResults.length === 0) return;
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedModalIndex = Math.min(selectedModalIndex + 1, modalSearchResults.length - 1);
            highlightModalResult(selectedModalIndex);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedModalIndex = Math.max(selectedModalIndex - 1, -1);
            if (selectedModalIndex >= 0) {
                highlightModalResult(selectedModalIndex);
            }
        } else if (e.key === 'Enter' && selectedModalIndex >= 0) {
            e.preventDefault();
            selectTickerFromModal(modalSearchResults[selectedModalIndex].symbol);
        }
    });
}

async function searchTickersModal(query) {
    try {
        const response = await fetch(`/api/market/search?q=${encodeURIComponent(query)}&limit=20`);
        const data = await response.json();
        
        const resultsDiv = document.getElementById('ticker-search-results-modal');
        modalSearchResults = data.results || [];
        selectedModalIndex = -1;
        
        if (modalSearchResults.length > 0) {
            resultsDiv.innerHTML = modalSearchResults.map((ticker, idx) => {
                const changeColor = (ticker.change_percent || 0) >= 0 ? 'var(--success-color)' : 'var(--error-color)';
                const changeSign = (ticker.change_percent || 0) >= 0 ? '+' : '';
                return `
                    <div class="search-result-item-modal ${idx === selectedModalIndex ? 'selected' : ''}" 
                         data-index="${idx}"
                         onclick="selectTickerFromModal('${ticker.symbol}')"
                         onmouseover="highlightModalResult(${idx})">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px;">
                            <div style="flex: 1;">
                                <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">${ticker.symbol}</div>
                                <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 2px;">
                                    ${ticker.name || ticker.symbol}
                                </div>
                                ${ticker.exchange ? `<div style="font-size: 11px; color: var(--text-secondary);">${ticker.exchange}</div>` : ''}
                            </div>
                            <div style="text-align: right; margin-left: 16px;">
                                ${ticker.current_price ? `<div style="font-weight: 600; font-size: 14px;">${formatCurrency(ticker.current_price)}</div>` : ''}
                                ${ticker.change_percent !== undefined ? `
                                    <div style="font-size: 12px; color: ${changeColor};">
                                        ${changeSign}${Math.abs(ticker.change_percent).toFixed(2)}%
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            resultsDiv.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-secondary);">No results found</div>';
        }
    } catch (error) {
        console.error('Modal search failed:', error);
        document.getElementById('ticker-search-results-modal').innerHTML = 
            '<div style="padding: 24px; text-align: center; color: var(--error-color);">Search error. Please try again.</div>';
    }
}

function highlightModalResult(index) {
    selectedModalIndex = index;
    document.querySelectorAll('.search-result-item-modal').forEach((item, idx) => {
        item.classList.toggle('selected', idx === index);
    });
}

async function selectTickerFromModal(symbol) {
    // Add ticker to trading list
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
            // Clear search input
            document.getElementById('ticker-search-input').value = '';
        } else {
            alert(data.message || 'Failed to add ticker');
        }
    } catch (error) {
        console.error('Failed to add ticker:', error);
        alert('Failed to add ticker. Please try again.');
    }
}

// Initialize when modal opens
document.addEventListener('DOMContentLoaded', () => {
    // Watch for modal opening
    const modal = document.getElementById('ticker-search-modal');
    if (modal) {
        const observer = new MutationObserver((mutations) => {
            if (modal.classList.contains('active')) {
                initTickerSearchModal();
                document.getElementById('ticker-search-input')?.focus();
            }
        });
        observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
    }
});
