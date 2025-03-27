import flask
from flask import make_response
from flask_cors import CORS
from diffusion import sample_glyphs
from PIL import Image
import numpy as np
app = flask.Flask(__name__)
CORS(app)

@app.route('/sample_diffusion')
def index():
    smpl = (sample_glyphs() * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
    img = Image.fromarray(smpl)
    response = make_response(flask.jsonify({'message': 'Hello, World!', 'image': img.tolist()}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)