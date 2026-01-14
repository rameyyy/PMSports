from flask import Blueprint, jsonify
from models.ncaamb_models import get_homepage_stats, get_pick_of_day_data

ncaamb_bp = Blueprint('ncaamb', __name__, url_prefix='/api/ncaamb')


@ncaamb_bp.route('/homepage-stats', methods=['GET'])
def homepage_stats():
    stats = get_homepage_stats()
    if stats is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(stats)


@ncaamb_bp.route('/pick-of-day', methods=['GET'])
def pick_of_day():
    data = get_pick_of_day_data()
    if data is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(data)
