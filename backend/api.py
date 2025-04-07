import flask
from flask import make_response, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import base64
from base64 import encodebytes
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
        self.eta = 1.0
        self.output = None
        self.input_images = None
        self.masks = None
        super().__init__()

    def run(self):
        latent_shape = (1, 26, 2048)
        diff_timestep = diff_model.ddpm.alphas.shape[0] - 1 # 1024
        times = torch.IntTensor(np.linspace(0, diff_timestep, diff_timestep+1, dtype=int)).to(device)
        z = torch.randn(latent_shape).to(device, dtype=dtype)
        with torch.no_grad():
            latent_input = diff_model.feature_to_latent(self.input_images)
            timesteps = list(range(diff_timestep, 0, -32))
            for idx, t in enumerate(tqdm(timesteps, desc='Sampling...')):
                t_curr = t
                t_prev = timesteps[idx+1] if idx+1 < len(timesteps) else 0

                abar_curr = diff_model.ddpm.alpha_bars[t_curr]
                abar_prev = diff_model.ddpm.alpha_bars[t_prev]

                predicted_noise = diff_model.predict_noise(z, times[t_curr:t_curr+1], self.label)
                # if self.cfg_coeff > 0:
                #     unconditional_predicted_noise = diff_model.predict_noise(z, t, None)
                #     predicted_noise = torch.lerp(predicted_noise, unconditional_predicted_noise, -self.cfg_coeff)

                pred_x0 = (z - predicted_noise * torch.sqrt(1 - abar_curr)) / torch.sqrt(abar_curr)
                var = (self.eta ** 2) * (1 - abar_curr / abar_prev) * (1 - abar_prev) / (1 - abar_curr)
                z_prev = torch.sqrt(abar_prev) * pred_x0 + torch.sqrt(1 - abar_prev - var) * predicted_noise + torch.sqrt(var) * torch.randn_like(pred_x0) * (t_prev > 0)
                z = z_prev * self.masks[:,:,None,None] + latent_input * (~self.masks)[:,:,None,None]

                self.progress += 1
            sample_glyphs = diff_model.latent_to_feature(z)
            self.output = sample_glyphs
            self.progress = "complete"

@app.route('/api/sample_diffusion', methods=['POST'])
def index():
    print("Received request")
    global global_threads
    global threads
    
    # Get the list of base64 encoded images from the request
    data = flask.request.get_json()
    selected_images = data.get('images', [])
    
    # Validate that we received exactly 26 images
    if len(selected_images) != 26:
        return make_response(jsonify({'error': 'Expected 26 images'}), 400)
    
    # Decode base64 images and convert to torch tensor
    masks = []
    decoded_images = []
    for img_b64 in selected_images:
        if img_b64 == True:
            img_array = np.ones((128, 128))
            masks.append(True)
        else:
            # Decode base64 string to image
            img_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_data)).convert('L')  # Convert to grayscale (1 channel)
            img_array = np.array(img).reshape((128, 128))
            masks.append(False)
            
        # Rescale from 0-255 to -1 to 1
        img_normalized = (img_array / 127.5) - 1.0
        decoded_images.append(img_normalized)
    
    # Convert list to torch tensor
    images_tensor = torch.tensor(np.array(decoded_images), dtype=dtype).to(device).unsqueeze(0)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).to(device).unsqueeze(0)
    
    threads[global_threads] = DiffusionThread()
    threads[global_threads].input_images = images_tensor  # Store the processed images
    threads[global_threads].masks = masks_tensor
    threads[global_threads].start()
    global_threads += 1

    response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/get_thread_progress/{global_threads-1}'}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/api/get_thread_progress/<int:thread_id>', methods=['GET'])
def get_thread_progress(thread_id):
    global threads

    if thread_id not in threads:
        return make_response(jsonify({'error': 'Thread not found'}), 404)
    if threads[thread_id].progress == "complete":
        smpl = (threads[thread_id].output * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
        response_pre = []
        for i in range(26):
            img_io = BytesIO()
            img = Image.fromarray(smpl[0, i]).convert('RGB')
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            response_pre.append(encodebytes(img_io.getvalue()).decode('ascii'))
        response = make_response(jsonify(response_pre))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    return make_response(jsonify({'progress': threads[thread_id].progress}))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)