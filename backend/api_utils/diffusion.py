import flask
from flask import make_response, jsonify, Blueprint
from flask_cors import cross_origin
from ml.ldm import LDM
from api_utils.config import device, diff_dtype, LOAD_MODELS, LOCAL_DEBUG, threads
import threading
import torch
import jwt
from api_utils.config import jwt_public_key, FontRunStage, NUM_GLYPHS

import sqlite3
import base64
from io import BytesIO
from base64 import encodebytes, decodebytes
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

diffusion_blueprint = Blueprint('diffusion', __name__)


if LOAD_MODELS:
    diff_model = LDM(diffusion_depth=1024, embedding_dim=2048, num_glyphs=26, label_dim=128, num_layers=24, num_heads=32, cond_dim=128)
    state_dict = torch.load('./models/ldm-basic-33928allchars_centered_scaled_sorted_filtered-128-128-0005-100-1300.pkl', map_location=torch.device('cpu'), weights_only=False)
    state_dict['enc_dec.z_min'] = state_dict['z_min'].min(dim=1)[0][0]
    state_dict['enc_dec.z_max'] = state_dict['z_max'].max(dim=1)[0][0]
    state_dict.pop('z_min')
    state_dict.pop('z_max')
    state_dict['ddpm.cond_embedding.weight'] = state_dict['ddpm.cond_embedding.weight'].repeat(1, 128)
    diff_model.load_state_dict(state_dict)
    diff_model = diff_model.to(device)
    diff_model.enc_dec = diff_model.enc_dec.to(dtype=diff_dtype)
    diff_model.ddpm = diff_model.ddpm.to(dtype=diff_dtype)
    diff_model = torch.compile(diff_model)
    diff_model.eval()

class DiffusionThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        self.label = None
        self.cfg_coeff = 0.0#3.0
        self.eta = 1.0
        self.output = None
        self.input_images = None
        self.masks = None
        self.email = None
        self.font_run_id = None
        self.prompt = "[No description provided]"
        super().__init__()

    def run(self):
        latent_shape = (1, 26, 2048)
        diff_timestep = diff_model.ddpm.alphas.shape[0] - 1 # 1024
        times = torch.IntTensor(np.linspace(0, diff_timestep, diff_timestep+1, dtype=int)).to(device)
        z = torch.randn(latent_shape).to(device, dtype=diff_dtype)
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
                
                noised_latent_input = diff_model.noise(latent_input, times[t_prev:t_prev+1])[0]
                z = z_prev * self.masks[:,:,None] + noised_latent_input * (~self.masks)[:,:,None]

                self.progress += 1
            sample_glyphs = diff_model.latent_to_feature(z)
            self.output = sample_glyphs * self.masks[:,:,None,None] + self.input_images * (~self.masks)[:,:,None,None]
            bitmap_images_numpy_to_db(self.email, self.font_run_id, self.output)
            self.progress = "complete"


#### DIFFUSION API ####

@diffusion_blueprint.route('/<string:font_run_id>/sample', methods=['POST'])
@cross_origin(supports_credentials=True)
def sample_diffusion(font_run_id):
    print("Received diffusion request for font run id: ", font_run_id)
    global threads

    auth_header = flask.request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        # valid auth token, get user id
        email = payload['email']

        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        cursor.execute('SELECT font_run_id, font_run_text, font_run_created_at, font_run_updated_at FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id))
        font_run = cursor.fetchone()
        conn.close()
        if font_run is None:
            return make_response(jsonify({'error': 'Font run not found'}), 404)
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)
    
    # Get the prompt from the payload
    data = flask.request.get_json()
    prompt = data.get('prompt', '[No description provided]')

    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE font_runs SET (font_run_text, font_run_stage, font_run_updated_at) = (?, ?, ?) WHERE (email, font_run_id) = (?, ?)', (prompt, FontRunStage.IMAGES_GENERATING.value, datetime.now(), email, font_run_id))
    conn.commit()
    conn.close()

    if LOCAL_DEBUG:
        bitmap_images_numpy_to_db(email, font_run_id, torch.zeros((1, NUM_GLYPHS, 128, 128), dtype=diff_dtype))
        # Update font stage
        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE font_runs SET (font_run_stage, font_run_updated_at) = (?, ?) WHERE (email, font_run_id) = (?, ?)', (FontRunStage.IMAGES_GENERATED.value, datetime.now(), email, font_run_id))
        conn.commit()
        conn.close()

        response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/diffusion/{font_run_id}/get_progress'}))
        # response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    
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
    images_tensor = torch.tensor(np.array(decoded_images), dtype=diff_dtype).to(device).unsqueeze(0)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).to(device).unsqueeze(0)
    
    ### TODO: mutex for global_threads
    threads[font_run_id] = {}
    threads[font_run_id]['diffusion'] = DiffusionThread()
    threads[font_run_id]['diffusion'].email = email
    threads[font_run_id]['diffusion'].font_run_id = font_run_id
    threads[font_run_id]['diffusion'].prompt = prompt
    threads[font_run_id]['diffusion'].input_images = images_tensor  # Store the processed images
    threads[font_run_id]['diffusion'].masks = masks_tensor
    threads[font_run_id]['diffusion'].start()

    response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/diffusion/{font_run_id}/get_progress'}))
    # response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@diffusion_blueprint.route('/<string:font_run_id>/get_progress', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_thread_progress_diffusion(font_run_id):
    global threads

    # authn and authz
    auth_header = flask.request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        # valid auth token, get user id
        email = payload['email']
        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        cursor.execute('SELECT font_run_id, font_run_text, font_run_created_at, font_run_updated_at FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id))
        font_run = cursor.fetchone()
        if font_run is None:
            return make_response(jsonify({'error': 'Font run not found'}), 404)
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)

    if LOCAL_DEBUG:
        smpl = bitmap_images_db_to_numpy(email, font_run_id)
        
        # Create response with loaded images
        response_pre = []
        for i in range(NUM_GLYPHS):
            img_io = BytesIO()
            img = Image.fromarray(smpl[0, i]).convert('RGB')
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            response_pre.append(encodebytes(img_io.getvalue()).decode('ascii'))
            
        response = make_response(jsonify(response_pre))
        # response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    if 'diffusion' not in threads or type(threads[font_run_id]['diffusion']) != DiffusionThread:
        return make_response(jsonify({'error': 'Thread not found'}), 404)
    if threads[font_run_id]['diffusion'].progress == "complete":
        smpl = bitmap_images_db_to_numpy(email, font_run_id)
        response_pre = []
        for i in range(NUM_GLYPHS):
            img_io = BytesIO()
            img = Image.fromarray(smpl[0, i]).convert('RGB')
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            response_pre.append(encodebytes(img_io.getvalue()).decode('ascii'))
        response = make_response(jsonify(response_pre))
        # response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    return make_response(jsonify({'progress': threads[font_run_id]['diffusion'].progress}))


def bitmap_images_numpy_to_db(email, font_run_id, output):
    '''
    output: (1, NUM_GLYPHS, 128, 128)
    '''
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    for i in range(NUM_GLYPHS):
        img_io = BytesIO()
        img = Image.fromarray((output[0, i] * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8))
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        data = encodebytes(img_io.getvalue()).decode('ascii')
        cursor.execute(f'UPDATE font_runs SET font_run_bitmap_images_{i} = ? WHERE (email, font_run_id) = (?, ?)', (data, email, font_run_id))
        conn.commit()
    conn.close()


def bitmap_images_db_to_numpy(email, font_run_id):
    '''
    return: (1, NUM_GLYPHS, 128, 128)
    '''
    smpl = np.zeros((1, NUM_GLYPHS, 128, 128), dtype=np.uint8)
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT {", ".join([f"font_run_bitmap_images_{i}" for i in range(NUM_GLYPHS)])} FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id))
    image_data = cursor.fetchone()
    conn.close()
    if image_data:
        for i in range(NUM_GLYPHS):
            if image_data[i] is not None:  # Check if image data exists
                try:
                    try:
                        img_bytes = base64.b64decode(image_data[i])
                        img_io = BytesIO(img_bytes)
                        img = Image.open(img_io).convert('L')
                        grayscale_img = np.array(img).reshape((128, 128)).astype(np.uint8)
                    except Exception as e:
                        print(f"Error decoding image {i}: {e}")
                        grayscale_img = np.zeros((128, 128), dtype=np.uint8) + 255
                    
                    # Store in smpl array
                    smpl[0, i] = grayscale_img
                except Exception as e:
                    print(f"Error loading image {i}: {e}")
                    # Use zero array if loading fails
                    smpl[0, i] = np.zeros((128, 128), dtype=np.uint8) + 255
            else:
                # Use zero array if no image data
                smpl[0, i] = np.zeros((128, 128), dtype=np.uint8) + 255
    return smpl