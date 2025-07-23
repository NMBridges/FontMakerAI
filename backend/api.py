import flask
from flask_cors import CORS

threads = {}
app = flask.Flask(__name__)
CORS(app, supports_credentials=True)

# Simple health check endpoints
@app.route('/api/health')
def api_health():
    """API health check endpoint"""
    return {'status': 'healthy'}, 200

from api_utils.auth import auth_blueprint
from api_utils.diffusion import diffusion_blueprint
from api_utils.vectorization import vectorization_blueprint
from api_utils.dashboard import dashboard_blueprint
from api_utils.fontrun import fontrun_blueprint

app.register_blueprint(auth_blueprint, url_prefix='/api/auth')
app.register_blueprint(diffusion_blueprint, url_prefix='/api/diffusion')
app.register_blueprint(vectorization_blueprint, url_prefix='/api/vectorization')
app.register_blueprint(dashboard_blueprint, url_prefix='/api/dashboard')
app.register_blueprint(fontrun_blueprint, url_prefix='/api/fontrun')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)