/**
 * Theme Management JavaScript
 */

// Load saved theme preference
function loadTheme() {
    const savedTheme = localStorage.getItem('dashboard-theme') || 'dark';
    document.body.className = `theme-${savedTheme}`;
}

// Save theme preference
function saveTheme(theme) {
    localStorage.setItem('dashboard-theme', theme);
}

// Toggle theme
function toggleTheme() {
    const body = document.body;
    if (body.classList.contains('theme-dark')) {
        body.classList.remove('theme-dark');
        body.classList.add('theme-light');
        saveTheme('light');
    } else {
        body.classList.remove('theme-light');
        body.classList.add('theme-dark');
        saveTheme('dark');
    }
}

// Load theme on page load
loadTheme();
