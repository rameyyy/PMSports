from flask import Blueprint, jsonify
from models.ncaamb_models import get_homepage_stats

ncaamb_bp = Blueprint('ncaamb', __name__, url_prefix='/api/ncaamb')


@ncaamb_bp.route('/homepage-stats', methods=['GET'])
def homepage_stats():
    stats = get_homepage_stats()
    if stats is None:
        return jsonify({'error': 'No data available'}), 404
    return jsonify(stats)
