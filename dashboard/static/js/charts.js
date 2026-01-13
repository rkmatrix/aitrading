/**
 * Charts JavaScript - TradingView Lightweight Charts Integration
 */

let tickerChart = null;

function initTickerChart(symbol, period = '1m') {
    const container = document.getElementById('ticker-chart-container');
    if (!container) return;
    
    // Clear existing chart
    if (tickerChart) {
        tickerChart.chart.remove();
        tickerChart = null;
    }
    
    // Show loading
    container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary);">Loading chart...</div>';
    
    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: 'var(--text-primary)',
            fontSize: 12,
        },
        grid: {
            vertLines: { color: 'var(--border-color)', style: 0 },
            horzLines: { color: 'var(--border-color)', style: 0 },
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            borderColor: 'var(--border-color)',
        },
        rightPriceScale: {
            borderColor: 'var(--border-color)',
        },
    });
    
    const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    });
    
    const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });
    
    tickerChart = { chart, series: candlestickSeries, volumeSeries };
    
    // Handle window resize
    const resizeObserver = new ResizeObserver(() => {
        if (tickerChart && container) {
            tickerChart.chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        }
    });
    resizeObserver.observe(container);
    
    // Load chart data
    loadTickerChartData(symbol, period);
}

async function loadTickerChartData(symbol, period = '1m') {
    try {
        const response = await fetch(`/api/ticker/${symbol}/chart-data?period=${period}`);
        const data = await response.json();
        
        if (tickerChart && data.candles && data.candles.length > 0) {
            // Convert data format
            const candles = data.candles.map(c => ({
                time: c.time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            }));
            
            const volumes = data.candles.map(c => ({
                time: c.time,
                value: c.volume,
                color: c.close >= c.open ? '#26a69a26' : '#ef535026',
            }));
            
            tickerChart.series.setData(candles);
            tickerChart.volumeSeries.setData(volumes);
            
            // Fit content
            tickerChart.chart.timeScale().fitContent();
        } else {
            console.warn('No chart data available');
        }
    } catch (error) {
        console.error('Failed to load chart data:', error);
        const container = document.getElementById('ticker-chart-container');
        if (container) {
            container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--error-color);">Failed to load chart data</div>';
        }
    }
}
