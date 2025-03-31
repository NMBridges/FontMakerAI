import flask
from flask import make_response, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from backend.ml.ldm import LDM

import os
print(os.getcwd())

diff_model = torch.load('models/ldm-basic-33928allchars_centered_scaled_sorted_filtered_(128,128)-0005-100-1400.pkl').to('cuda', dtype=torch.float32)
font_model = torch.load('models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl').to('cuda', dtype=torch.bfloat16)

app = flask.Flask(__name__)
CORS(app)

@app.route('/api/sample_diffusion')
def index():
    img_io = BytesIO()

    latent_shape = (1, 26, 2048)
    sample_glyphs = diff_model.sample(latent_shape)

    smpl = (sample_glyphs * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
    img = Image.fromarray(smpl)
    img.save(img_io, format='JPEG')
    img_io.seek(0)

    response = make_response(send_file(img_io, mimetype='image/jpeg'))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)