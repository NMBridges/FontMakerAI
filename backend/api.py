import flask
from flask import make_response, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import base64
import os
from base64 import encodebytes
import threading
from tqdm import tqdm
from ml.ldm import LDM
from ml.fontmodel import FontModel
from ml.tokenizer import Tokenizer
from parsing.tablelist_utils import numbers_first, make_non_cumulative
from parsing.glyph_viz import Visualizer
from config import operators, DecodeInstruction, DecodeType, SamplingType

import sys
sys.path.insert(0, './ml')

device = 'cuda'
diff_dtype = torch.float32
path_dtype = torch.float16
global_threads = 0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading models")

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

font_model = torch.load('./models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', weights_only=False).to('cuda', dtype=path_dtype)
# font_model = torch.quantization.quantize_dynamic(
#     font_model, {torch.nn.Linear}, dtype=path_dtype
# )
font_model.eval()

print("Loaded models")

pad_token = "<PAD>"
sos_token = "<SOS>"
eos_token = "<EOS>"
tokenizer = Tokenizer(
    min_number=-500,
    max_number=500,
    possible_operators=operators,
    pad_token=pad_token,
    sos_token=sos_token,
    eos_token=eos_token
)


threads = {}
app = flask.Flask(__name__)
CORS(app)





#### THREADS ####

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
            self.progress = "complete"


def numeric_tokens_to_im(sequence, decode_instr, done=False):
    if not done:
        toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence]
    else:
        if len(sequence) == decode_instr.max_seq_len:
            toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence] + ['endchar']
        else:
            toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

    toks = [tok for tok in toks if tok != '<PAD2>' and tok != '<PAD>']
    toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
    viz = Visualizer(toks)
    
    im_pixel_size = (128, 128)
    crop_factor = 1
    dpi = 1
    boundaries = (int((im_pixel_size[0] * (crop_factor * 100 / dpi - 1)) // 2), int((im_pixel_size[1] * (crop_factor * 100 / dpi - 1)) // 2))
    im_size_inches = ((im_pixel_size[0] * crop_factor) / dpi, (im_pixel_size[1] * crop_factor) / dpi)
    img_arr = viz.draw(
        display=False,
        filename=None,
        return_image=True,
        center=False,
        im_size_inches=im_size_inches,
        bounds=(-300, 300),
        dpi=dpi
    )[:,:,0]

    return img_arr

class PathThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        self.output = None
        self.image = None
        self.decode_instr = None
        self.log_file = None
        super().__init__()

    def run(self):

        im = self.image.unsqueeze(1)
        with torch.no_grad():
            sequence = font_model.decode(im, None, self.decode_instr, self.log_file)[0].cpu().detach().numpy().flatten()

        img_arr = numeric_tokens_to_im(sequence, self.decode_instr)
            
        self.progress = "complete"
        self.output = img_arr




#### DIFFUSION API ####

@app.route('/api/sample_diffusion', methods=['POST'])
def sample_diffusion():
    print("Received diffusion request")
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
    images_tensor = torch.tensor(np.array(decoded_images), dtype=diff_dtype).to(device).unsqueeze(0)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).to(device).unsqueeze(0)
    
    ### TODO: mutex for global_threads
    thread_id = global_threads
    global_threads += 1
    threads[thread_id] = DiffusionThread()
    threads[thread_id].input_images = images_tensor  # Store the processed images
    threads[thread_id].masks = masks_tensor
    threads[thread_id].start()

    response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/get_thread_progress_diffusion/{thread_id}'}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/api/get_thread_progress_diffusion/<int:thread_id>', methods=['GET'])
def get_thread_progress_diffusion(thread_id):
    global threads

    if thread_id not in threads or type(threads[thread_id]) != DiffusionThread:
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




#### PATH API ####

@app.route('/api/sample_path', methods=['POST'])
def sample_path():
    print("Received path request")
    global global_threads
    global threads

    data = flask.request.get_json()
    image = data.get('image', None)
    if image is None:
        return make_response(jsonify({'error': 'Expected image'}), 400)
    
    img_data = base64.b64decode(image)
    image = Image.open(BytesIO(img_data)).convert('L')  # Convert to grayscale (1 channel)
    image = np.array(image).reshape((1, 128, 128))
    image = (image / 127.5) - 1.0
    image = torch.tensor(image, dtype=path_dtype).to(device)

    decode_instr = DecodeInstruction( # NOTE: doesn't matter unless loading from .config.txt fails
        DecodeType.ANCESTRAL,
        SamplingType.GREEDY,
        max_seq_len=5040,
        k=5,
        p=0,
        temp=0,
        beam_size=6,
    )
    with open('./.config.txt', 'r') as cf:
        lines = cf.readlines()
        if len(lines) != 7:
            print(f"Not decoding this iteration; .config.txt has wrong number of lines ({len(lines)})")
            return
        else:
            decode_instr = DecodeInstruction(
                decode_type=DecodeType[lines[0].split("=")[-1].split(".")[-1].strip()],
                sampling_type=SamplingType[lines[1].split("=")[-1].split(".")[-1].strip()],
                max_seq_len=int(lines[2].split("=")[-1].strip()),
                k=int(lines[3].split("=")[-1].strip()),
                p=float(lines[4].split("=")[-1].strip()),
                temp=float(lines[5].split("=")[-1].strip()),
                beam_size=int(lines[6].split("=")[-1].strip())
            )
    
    ### TODO: mutex for global_threads
    thread_id = global_threads
    global_threads += 1
    threads[thread_id] = PathThread()
    threads[thread_id].image = image
    threads[thread_id].decode_instr = decode_instr
    threads[thread_id].log_file = f"{thread_id}.log"
    threads[thread_id].start()

    response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/get_thread_progress_path/{thread_id}'}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response
    
@app.route('/api/get_thread_progress_path/<int:thread_id>', methods=['GET'])
def get_thread_progress_path(thread_id):
    global threads

    if thread_id not in threads or type(threads[thread_id]) != PathThread:
        return make_response(jsonify({'error': 'Thread not found'}), 404)
    
    if threads[thread_id].progress == "complete":
        im = threads[thread_id].output
        img_io = BytesIO()
        img = Image.fromarray(im.astype(np.uint8))
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        response_pre = encodebytes(img_io.getvalue()).decode('ascii')
        response = make_response(jsonify([response_pre]))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    else:
        # Check if there's a progress log file for this thread
        log_file_path = f"{thread_id}.log"
        
        tok_seq = None
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as log_file:
                    tok_seq = log_file.read().strip()
            except Exception as e:
                print(f"Error reading progress log file: {e}")
        if not tok_seq or len(tok_seq.split(" ")) <= 7:
            return make_response(jsonify({'progress': threads[thread_id].progress}))
        else:
            im = numeric_tokens_to_im(torch.tensor(list(map(int, tok_seq.split(" "))), dtype=torch.int32), threads[thread_id].decode_instr)
            img_io = BytesIO()
            img = Image.fromarray(im.astype(np.uint8))
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            response_pre = encodebytes(img_io.getvalue()).decode('ascii')
            response = make_response(jsonify({'image': [response_pre], 'progress': len(tok_seq.split(" ")) / threads[thread_id].decode_instr.max_seq_len}))
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)