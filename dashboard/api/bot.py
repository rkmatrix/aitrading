"""
Bot Control API Endpoints
"""
from flask import Blueprint, jsonify, request
from dashboard.bot_controller import BotController

bp = Blueprint('bot', __name__)
controller = BotController()


@bp.route('/status', methods=['GET'])
def get_status():
    """Get bot status."""
    status = controller.get_status()
    return jsonify(status)


@bp.route('/start', methods=['POST'])
def start_bot():
    """Start the bot."""
    result = controller.start()
    return jsonify(result)


@bp.route('/stop', methods=['POST'])
def stop_bot():
    """Stop the bot."""
    result = controller.stop()
    return jsonify(result)


@bp.route('/kill-switch', methods=['POST'])
def toggle_kill_switch():
    """Toggle kill switch."""
    data = request.get_json() or {}
    activate = data.get('activate', True)
    result = controller.toggle_kill_switch(activate)
    return jsonify(result)
