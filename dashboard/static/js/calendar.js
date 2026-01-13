/**
 * Calendar Widget JavaScript
 */

let currentCalendarDate = new Date();
let calendarActivity = {};

function initCalendar() {
    const prevBtn = document.getElementById('calendar-prev-month');
    const nextBtn = document.getElementById('calendar-next-month');
    
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            currentCalendarDate.setMonth(currentCalendarDate.getMonth() - 1);
            loadCalendar();
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            currentCalendarDate.setMonth(currentCalendarDate.getMonth() + 1);
            loadCalendar();
        });
    }
    
    loadCalendar();
}

async function loadCalendar() {
    const year = currentCalendarDate.getFullYear();
    const month = currentCalendarDate.getMonth() + 1;
    
    try {
        const response = await fetch(`/api/calendar/activity?year=${year}&month=${month}`);
        const data = await response.json();
        
        calendarActivity = data.activity || {};
        renderCalendar(year, month);
    } catch (error) {
        console.error('Failed to load calendar:', error);
    }
}

function renderCalendar(year, month) {
    const grid = document.getElementById('calendar-grid');
    const monthYear = document.getElementById('calendar-month-year');
    
    if (!grid || !monthYear) return;
    
    // Update month/year header
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December'];
    monthYear.textContent = `${monthNames[month - 1]} ${year}`;
    
    // Get first day of month and number of days
    const firstDay = new Date(year, month - 1, 1);
    const lastDay = new Date(year, month, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();
    
    // Day names
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    
    let html = '<div class="calendar-weekdays">';
    dayNames.forEach(day => {
        html += `<div class="calendar-weekday">${day}</div>`;
    });
    html += '</div>';
    
    html += '<div class="calendar-days">';
    
    // Empty cells for days before month starts
    for (let i = 0; i < startingDayOfWeek; i++) {
        html += '<div class="calendar-day empty"></div>';
    }
    
    // Days of the month
    const today = new Date();
    for (let day = 1; day <= daysInMonth; day++) {
        const date = new Date(year, month - 1, day);
        const dateStr = date.toISOString().split('T')[0];
        const activity = calendarActivity[dateStr];
        
        const isToday = date.toDateString() === today.toDateString();
        let dayClass = 'calendar-day';
        let bgColor = 'transparent';
        let borderColor = 'transparent';
        
        if (activity) {
            if (activity.is_profitable) {
                bgColor = 'rgba(38, 166, 154, 0.2)';
                borderColor = 'var(--success-color)';
            } else if (activity.total_pnl < 0) {
                bgColor = 'rgba(239, 83, 80, 0.2)';
                borderColor = 'var(--error-color)';
            } else {
                bgColor = 'rgba(255, 183, 77, 0.2)';
                borderColor = 'var(--warning-color)';
            }
        }
        
        if (isToday) {
            dayClass += ' today';
        }
        
        html += `
            <div class="${dayClass}" 
                 data-date="${dateStr}"
                 style="background-color: ${bgColor}; border-color: ${borderColor};"
                 onclick="showDayDetails('${dateStr}')">
                <div class="calendar-day-number">${day}</div>
                ${activity ? `
                    <div class="calendar-day-stats">
                        <div style="font-size: 10px; color: var(--text-secondary);">
                            ${activity.trade_count} trade${activity.trade_count !== 1 ? 's' : ''}
                        </div>
                        <div style="font-size: 10px; font-weight: 600; color: ${activity.is_profitable ? 'var(--success-color)' : 'var(--error-color)'};">
                            ${formatCurrency(activity.total_pnl)}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    html += '</div>';
    
    grid.innerHTML = html;
}

async function showDayDetails(dateStr) {
    try {
        const response = await fetch(`/api/calendar/day/${dateStr}`);
        const data = await response.json();
        
        // Create modal or update existing one
        const modal = document.getElementById('day-details-modal') || createDayDetailsModal();
        modal.classList.add('active');
        
        // Populate modal content
        const date = new Date(dateStr);
        document.getElementById('day-details-title').textContent = 
            date.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        
        const statsDiv = document.getElementById('day-details-stats');
        statsDiv.innerHTML = `
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">${data.trade_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Filled</div>
                <div class="stat-value">${data.filled_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value" style="color: ${data.total_pnl >= 0 ? 'var(--success-color)' : 'var(--error-color)'}">
                    ${formatCurrency(data.total_pnl)}
                </div>
            </div>
        `;
        
        // Populate trades table
        const tradesTbody = document.getElementById('day-details-trades-tbody');
        if (data.trades && data.trades.length > 0) {
            tradesTbody.innerHTML = data.trades.map(trade => `
                <tr>
                    <td>${formatTime(trade.timestamp)}</td>
                    <td><strong>${trade.symbol}</strong></td>
                    <td style="color: ${trade.side === 'BUY' ? 'var(--success-color)' : 'var(--error-color)'}">${trade.side}</td>
                    <td>${trade.qty}</td>
                    <td>${formatCurrency(trade.price)}</td>
                    <td><span class="status-badge ${trade.status.toLowerCase()}">${trade.status}</span></td>
                </tr>
            `).join('');
        } else {
            tradesTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 24px;">No trades on this day</td></tr>';
        }
        
    } catch (error) {
        console.error('Failed to load day details:', error);
    }
}

function createDayDetailsModal() {
    const modal = document.createElement('div');
    modal.id = 'day-details-modal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 800px;">
            <div class="modal-header">
                <h2 class="modal-title" id="day-details-title">Day Details</h2>
                <button class="modal-close" onclick="closeModal('day-details-modal')">×</button>
            </div>
            <div id="day-details-stats" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;"></div>
            <h3 style="margin-bottom: 12px;">Trades</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="day-details-trades-tbody"></tbody>
            </table>
        </div>
    `;
    document.body.appendChild(modal);
    return modal;
}

// Initialize calendar when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCalendar);
} else {
    initCalendar();
}
