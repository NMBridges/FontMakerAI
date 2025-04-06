import flask
from flask import make_response, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import threading
from tqdm import tqdm
from ml.ldm import LDM
from ml.fontmodel import FontModel

import sys
sys.path.insert(0, './ml')
device = 'cuda'
dtype = torch.float32
global_threads = 0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

diff_model = LDM(diffusion_depth=1024, embedding_dim=2048, num_glyphs=26, label_dim=128, num_layers=24, num_heads=32, cond_dim=128)

state_dict = torch.load('./models/ldm-basic-33928allchars_centered_scaled_sorted_filtered-128-128-0005-100-1300.pkl', map_location=torch.device('cpu'), weights_only=False)
# print([(x[0], x[1].shape) for x in state_dict.items()])
state_dict['enc_dec.z_min'] = state_dict['z_min'].min(dim=1)[0][0]
state_dict['enc_dec.z_max'] = state_dict['z_max'].max(dim=1)[0][0]
state_dict.pop('z_min')
state_dict.pop('z_max')
state_dict['ddpm.cond_embedding.weight'] = state_dict['ddpm.cond_embedding.weight'].repeat(1, 128)
diff_model.load_state_dict(state_dict)
diff_model = diff_model.to(device)
diff_model.enc_dec = diff_model.enc_dec.to(dtype=dtype)
diff_model.ddpm = diff_model.ddpm.to(dtype=dtype)
diff_model = torch.compile(diff_model)
diff_model.eval()

# font_model = torch.load('./models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', weights_only=False).to('cuda', dtype=torch.bfloat16)

threads = {}
app = flask.Flask(__name__)
CORS(app)

class DiffusionThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        self.label = None
        self.cfg_coeff = 0.0#3.0
        self.gamma = 1.0
        self.output = None
        super().__init__()

    def run(self):
        latent_shape = (1, 26, 2048)
        diff_timestep = diff_model.ddpm.alphas.shape[0] - 1
        times = torch.IntTensor(np.linspace(0, diff_timestep, diff_timestep+1, dtype=int)).to(device)
        z = torch.randn(latent_shape).to(device, dtype=dtype)
        with torch.no_grad():
            timesteps = list(range(diff_timestep, 0, -1))# + [1]
            for i, t in enumerate(tqdm(timesteps, desc='Sampling...')):
                t_curr = t
                t_prev = timesteps[i-1] if i > 0 else 0

                predicted_noise = diff_model.predict_noise(z, times[t_curr:t_curr+1], self.label)
                # if self.cfg_coeff > 0:
                #     unconditional_predicted_noise = diff_model.predict_noise(z, t, None)
                #     predicted_noise = torch.lerp(predicted_noise, unconditional_predicted_noise, -self.cfg_coeff)

                pred_x0 = (z - predicted_noise * torch.sqrt(1 - diff_model.ddpm.alpha_bars[t_curr])) / torch.sqrt(diff_model.ddpm.alpha_bars[t_curr])
                var = (self.gamma ** 2) * (1 - diff_model.ddpm.alpha_bars[t_curr] / diff_model.ddpm.alpha_bars[t_prev]) * (1 - diff_model.ddpm.alpha_bars[t_prev]) / (1 - diff_model.ddpm.alpha_bars[t_curr])
                z_prev = torch.sqrt(diff_model.ddpm.alpha_bars[t_prev]) * pred_x0 + torch.sqrt(1 - diff_model.ddpm.alpha_bars[t_prev] - var) * predicted_noise + torch.sqrt(var) * torch.randn_like(pred_x0)
                z = z_prev

                self.progress += 1
            sample_glyphs = diff_model.latent_to_feature(z)
            self.output = sample_glyphs
            self.progress = "complete"

@app.route('/api/sample_diffusion')
def index():
    print("Received request")
    global global_threads
    global threads

    threads[global_threads] = DiffusionThread()
    threads[global_threads].start()
    global_threads += 1

    response = make_response(jsonify({'progress': 0}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/api/sample_diffusion_thread/<int:thread_id>')
def sample_diffusion_thread(thread_id):
    global threads

    if thread_id not in threads:
        return make_response(jsonify({'error': 'Thread not found'}), 404)
    if threads[thread_id].progress == "complete":
        img_io = BytesIO()
        smpl = (threads[thread_id].output * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
        # Reshape the sample from (1,26,128,128) to a grid of (128*6, 128*5)
        # This creates a 6x5 grid with 4 blank tiles at the bottom
        grid_height, grid_width = 6, 5
        tile_size = 128
        grid_img = np.zeros((grid_height * tile_size, grid_width * tile_size), dtype=np.uint8)
        
        # Fill the grid with the 26 glyphs (leaving 4 blank tiles at the end)
        for i in range(26):
            row = i // grid_width
            col = i % grid_width
            grid_img[row * tile_size:(row + 1) * tile_size, 
                    col * tile_size:(col + 1) * tile_size] = smpl[0, i]
        
        # Replace the original sample with our grid
        smpl = grid_img
        img = Image.fromarray(smpl).convert('RGB')
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        response = make_response(send_file(img_io, mimetype='image/jpeg'))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    return make_response(jsonify({'progress': threads[thread_id].progress}))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)