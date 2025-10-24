from flask import Flask
# from flask_cors import CORS
from routes.ufc_routes import ufc_bp
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

# Register blueprints
app.register_blueprint(ufc_bp)

@app.route('/')
def index():
    return {'message': 'AlgoPicks API is running'}

@app.route('/health')
def health():
    return {'status': 'healthy'}

# ADD THIS TO SEE ALL ROUTES
@app.route('/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return {'routes': routes}

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
