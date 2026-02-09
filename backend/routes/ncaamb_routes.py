from flask import Blueprint, jsonify, request
from models.ncaamb_models import get_homepage_stats, get_games_by_date, get_model_performance

ncaamb_bp = Blueprint('ncaamb', __name__, url_prefix='/api/ncaamb')


@ncaamb_bp.route('/homepage-stats', methods=['GET'])
def homepage_stats():
    stats = get_homepage_stats()
    if stats is None:
        return jsonify({'error': 'No data available'}), 404
    return jsonify(stats)


@ncaamb_bp.route('/games', methods=['GET'])
def games():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'date query parameter is required'}), 400
    games = get_games_by_date(date_str)
    return jsonify({'date': date_str, 'games': games, 'count': len(games)})


@ncaamb_bp.route('/performance', methods=['GET'])
def performance():
    data = get_model_performance()
    if data is None:
        return jsonify({'error': 'No performance data available'}), 404
    return jsonify(data)
