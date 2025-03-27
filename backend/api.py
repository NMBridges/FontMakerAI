import flask
from flask import make_response, send_file
from flask_cors import CORS
# from diffusion import sample_glyphs
from PIL import Image
from io import BytesIO
import numpy as np
app = flask.Flask(__name__)
CORS(app)

@app.route('/api/sample_diffusion')
def index():
    # smpl = (sample_glyphs() * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
    img_io = BytesIO()
    smpl = np.zeros((256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(smpl)
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    response = make_response(send_file(img_io, mimetype='image/jpeg'))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)