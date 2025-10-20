from flask import Blueprint, jsonify, request
from models.ufc_models import (
    get_all_events,
    get_upcoming_events,
    get_past_events,
    get_event_by_id,
    get_fights_by_event,
    get_model_stats,
    get_fight_odds,
    get_calibrated_probability,
    get_fighter_stats
)

# CREATE THE BLUEPRINT FIRST
ufc_bp = Blueprint('ufc', __name__, url_prefix='/api/ufc')

@ufc_bp.route('/events', methods=['GET'])
def get_events():
    """Get all UFC events"""
    events = get_all_events()
    if events is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(events)

@ufc_bp.route('/events/upcoming', methods=['GET'])
def get_upcoming():
    """Get upcoming UFC events"""
    events = get_upcoming_events()
    if events is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(events)

@ufc_bp.route('/events/past', methods=['GET'])
def get_past():
    """Get past UFC events"""
    events = get_past_events()
    if events is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(events)

@ufc_bp.route('/events/<event_id>', methods=['GET'])
def get_event(event_id):
    """Get a specific event"""
    event = get_event_by_id(event_id)
    if event is None:
        return jsonify({'error': 'Event not found'}), 404
    return jsonify(event)

@ufc_bp.route('/events/<event_id>/fights', methods=['GET'])
def get_fights(event_id):
    """Get fights for a specific event"""
    fights = get_fights_by_event(event_id)
    if fights is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(fights)

@ufc_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get model performance statistics"""
    stats = get_model_stats()
    if stats is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(stats)

@ufc_bp.route('/fights/<fight_id>/odds', methods=['GET'])
def get_odds(fight_id):
    """Get bookmaker odds for a fight"""
    odds = get_fight_odds(fight_id)
    if odds is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(odds)

@ufc_bp.route('/fighters/<fighter_id>', methods=['GET'])
def get_fighter(fighter_id):
    """Get fighter details"""
    fighter = get_fighter_stats(fighter_id)
    if fighter is None:
        return jsonify({'error': 'Fighter not found'}), 404
    return jsonify(fighter)

@ufc_bp.route('/calibrate', methods=['POST'])
def calibrate_probability():
    """Get calibrated probability for a model prediction"""
    data = request.json
    model_name = data.get('model_name')
    raw_prob = data.get('probability')
    
    calibrated = get_calibrated_probability(model_name, raw_prob)
    return jsonify({
        'raw_probability': raw_prob,
        'calibrated_probability': calibrated
    })