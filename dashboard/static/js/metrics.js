/**
 * Metrics JavaScript
 */

// Metrics update functions are in main.js
// This file can contain chart rendering logic

let equityChart = null;

function initEquityChart() {
    // Initialize TradingView Lightweight Chart for equity curve
    const chartContainer = document.getElementById('main-chart');
    if (!chartContainer) return;
    
    const chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: {
            background: { color: 'transparent' },
            textColor: 'var(--text-primary)',
        },
        grid: {
            vertLines: { color: 'var(--border-color)' },
            horzLines: { color: 'var(--border-color)' },
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
        },
    });
    
    const lineSeries = chart.addLineSeries({
        color: 'var(--accent-color)',
        lineWidth: 2,
    });
    
    equityChart = { chart, lineSeries };
}

function updateEquityChart(data) {
    if (!equityChart) return;
    
    // Update chart with new data points
    equityChart.lineSeries.setData(data);
}

// Initialize chart when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEquityChart);
} else {
    initEquityChart();
}
