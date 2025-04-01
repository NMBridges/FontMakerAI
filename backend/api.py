import flask
from flask import make_response, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from backend.ml.ldm import LDM
from backend.ml.fontmodel import FontModel

import sys
sys.path.insert(0, './backend/ml')

diff_model = LDM(diffusion_depth=1024, embedding_dim=2048, num_glyphs=26, label_dim=128, num_layers=24, num_heads=32, cond_dim=128).to('cuda', dtype=torch.float16)

state_dict = torch.load('./backend/models/ldm-basic-33928allchars_centered_scaled_sorted_filtered-128-128-0005-100-1300.pkl', weights_only=False)
# print([(x[0], x[1].shape) for x in state_dict.items()])
state_dict['enc_dec.z_min'] = state_dict['z_min'].min(dim=1)[0][0]
state_dict['enc_dec.z_max'] = state_dict['z_max'].max(dim=1)[0][0]
state_dict.pop('z_min')
state_dict.pop('z_max')
state_dict['ddpm.cond_embedding.weight'] = state_dict['ddpm.cond_embedding.weight'].repeat(1, 128)
diff_model.load_state_dict(state_dict)
diff_model = diff_model.to('cuda', dtype=torch.float32)

# font_model = torch.load('./backend/models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', weights_only=False).to('cuda', dtype=torch.bfloat16)

app = flask.Flask(__name__)
CORS(app)

@app.route('/api/sample_diffusion')
def index():
    print("Received request")
    img_io = BytesIO()

    latent_shape = (1, 26, 2048)
    sample_glyphs = diff_model.sample(latent_shape)
    smpl = (sample_glyphs * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
    
    img = Image.fromarray(smpl[0,0]).convert('RGB')
    # img = Image.fromarray(smpl)
    img.save(img_io, format='JPEG')
    img_io.seek(0)

    response = make_response(send_file(img_io, mimetype='image/jpeg'))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)