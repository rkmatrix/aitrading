/**
 * Bot Control JavaScript
 */

let botStartTime = null;

function updateUptime() {
    const uptimeEl = document.getElementById('bot-uptime');
    if (!uptimeEl) return;
    
    if (dashboardState.botStatus === 'running') {
        if (!botStartTime) {
            botStartTime = Date.now();
        }
        
        const elapsed = Math.floor((Date.now() - botStartTime) / 1000);
        const hours = Math.floor(elapsed / 3600);
        const minutes = Math.floor((elapsed % 3600) / 60);
        const seconds = elapsed % 60;
        
        uptimeEl.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    } else {
        botStartTime = null;
        uptimeEl.textContent = '00:00:00';
    }
}

// Update uptime every second
setInterval(updateUptime, 1000);
