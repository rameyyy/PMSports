from flask import Blueprint, jsonify, request
from models.ufc_models import (
    get_all_events,
    get_upcoming_events,
    get_past_events,
    get_past_events_total_count,
    get_event_by_id,
    get_fights_by_event,
    get_fight_odds,
    get_fight_prediction,
    get_recent_predictions,
    get_predictions_by_confidence,
    get_fighter_stats,
    get_model_accuracies,
    get_all_bets,
    get_all_bets_total_count,
    get_pending_bets,
    get_pending_bets_total_count,
    get_settled_bets,
    get_settled_bets_total_count,
    get_betting_stats,
    get_risk_metrics,
    get_bet_analytics_by_strategy
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

@ufc_bp.route('/bets', methods=['GET'])
def get_bets():
    """Get all user bets with server-side pagination (paginated by events, not bets)"""
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10, type=int)

    # Validate pagination parameters
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:  # Cap at 100 to prevent abuse
        limit = 10

    # Calculate offset from page number
    offset = (page - 1) * limit

    # Fetch paginated bets from database (grouped by event)
    bets = get_all_bets(offset=offset, limit=limit)
    if bets is None:
        return jsonify({'error': 'Database error'}), 500

    # Get total count of distinct events for pagination metadata
    total = get_all_bets_total_count()

    return jsonify({
        'bets': bets,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit  # Ceiling division
        }
    })

@ufc_bp.route('/bets/pending', methods=['GET'])
def get_pending():
    """Get pending bets with server-side pagination (paginated by events, not bets)"""
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10, type=int)

    # Validate pagination parameters
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:  # Cap at 100 to prevent abuse
        limit = 10

    # Calculate offset from page number
    offset = (page - 1) * limit

    # Fetch paginated bets from database (grouped by event)
    bets = get_pending_bets(offset=offset, limit=limit)
    if bets is None:
        return jsonify({'error': 'Database error'}), 500

    # Get total count of distinct events for pagination metadata
    total = get_pending_bets_total_count()

    return jsonify({
        'bets': bets,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit  # Ceiling division
        }
    })

@ufc_bp.route('/bets/settled', methods=['GET'])
def get_settled():
    """Get settled bets with server-side pagination (paginated by events, not bets)"""
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10, type=int)

    # Validate pagination parameters
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:  # Cap at 100 to prevent abuse
        limit = 10

    # Calculate offset from page number
    offset = (page - 1) * limit

    # Fetch paginated bets from database (grouped by event)
    bets = get_settled_bets(offset=offset, limit=limit)
    if bets is None:
        return jsonify({'error': 'Database error'}), 500

    # Get total count of distinct events for pagination metadata
    total = get_settled_bets_total_count()

    return jsonify({
        'bets': bets,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit  # Ceiling division
        }
    })

@ufc_bp.route('/bets/stats', methods=['GET'])
def get_bet_stats():
    """Get betting statistics"""
    stats = get_betting_stats()
    if stats is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(stats)

@ufc_bp.route('/model-accuracies', methods=['GET'])
def get_accuracies():
    """Get model accuracy statistics"""
    accuracies = get_model_accuracies()
    if accuracies is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(accuracies)

@ufc_bp.route('/events/past', methods=['GET'])
def get_past():
    """Get past UFC events with server-side pagination"""
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10, type=int)

    # Validate pagination parameters
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:  # Cap at 100 to prevent abuse
        limit = 10

    # Calculate offset from page number
    offset = (page - 1) * limit

    # Fetch paginated events from database
    events = get_past_events(offset=offset, limit=limit)
    if events is None:
        return jsonify({'error': 'Database error'}), 500

    # Get total count for pagination metadata
    total = get_past_events_total_count()

    return jsonify({
        'events': events,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit  # Ceiling division
        }
    })

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

@ufc_bp.route('/fights/<fight_id>/prediction', methods=['GET'])
def get_prediction(fight_id):
    """Get the prediction for a specific fight"""
    prediction = get_fight_prediction(fight_id)
    if prediction is None:
        return jsonify({'error': 'Prediction not found'}), 404
    return jsonify(prediction)

@ufc_bp.route('/predictions/recent', methods=['GET'])
def get_recent():
    """Get recent predictions"""
    limit = request.args.get('limit', default=10, type=int)
    predictions = get_recent_predictions(limit)
    if predictions is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(predictions)

@ufc_bp.route('/predictions/high-confidence', methods=['GET'])
def get_high_confidence():
    """Get high confidence predictions"""
    min_confidence = request.args.get('min_confidence', default=60.0, type=float)
    limit = request.args.get('limit', default=20, type=int)
    predictions = get_predictions_by_confidence(min_confidence, limit)
    if predictions is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(predictions)

@ufc_bp.route('/fighters/<fighter_id>', methods=['GET'])
def get_fighter(fighter_id):
    """Get fighter details"""
    fighter = get_fighter_stats(fighter_id)
    if fighter is None:
        return jsonify({'error': 'Fighter not found'}), 404
    return jsonify(fighter)

@ufc_bp.route('/fights/<fight_id>/odds', methods=['GET'])
def get_odds(fight_id):
    """Get bookmaker odds for a fight with EV calculations"""
    odds = get_fight_odds(fight_id)
    if odds is None:
        return jsonify({'error': 'Database error'}), 500

    # Transform the data to include EV and format properly
    formatted_odds = []
    for odd in odds:
        formatted_odds.append({
            'fight_id': odd['fight_id'],
            'bookmaker': odd['bookmaker'],
            'fighter1_id': odd['fighter1_id'],
            'fighter2_id': odd['fighter2_id'],
            'fighter1_odds': odd['fighter1_odds'],
            'fighter2_odds': odd['fighter2_odds'],
            'fighter1_odds_percent': round(float(odd['fighter1_odds_percent']), 2),
            'fighter2_odds_percent': round(float(odd['fighter2_odds_percent']), 2),
            'fighter1_ev': round(float(odd['fighter1_ev']), 2) if odd.get('fighter1_ev') is not None else None,
            'fighter2_ev': round(float(odd['fighter2_ev']), 2) if odd.get('fighter2_ev') is not None else None,
            'vigor': round(float(odd['vigor']), 2),
            'algopick_prediction': odd['algopick_prediction'],
            'algopick_probability': round(float(odd['algopick_probability']), 2) if odd.get('algopick_probability') is not None else None
        })

    return jsonify(formatted_odds)

@ufc_bp.route('/risk-metrics', methods=['GET'])
def get_risk_metrics_route():
    """Get all risk metrics from Kelly analytics"""
    risk_metrics = get_risk_metrics()
    if risk_metrics is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(risk_metrics)

@ufc_bp.route('/bet-analytics/<strategy_name>', methods=['GET'])
def get_bet_analytics_route(strategy_name):
    """Get bet analytics for a specific strategy (e.g., Kelly_4pct, Flat_50, Kelly_5pct)"""
    bet_analytics = get_bet_analytics_by_strategy(strategy_name)
    if bet_analytics is None:
        return jsonify({'error': 'Database error'}), 500
    if len(bet_analytics) == 0:
        return jsonify({'error': f'No data found for strategy: {strategy_name}'}), 404
    return jsonify(bet_analytics)