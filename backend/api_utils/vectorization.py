import threading
import torch
import flask
from flask import make_response, jsonify, Blueprint
from flask_cors import cross_origin
import jwt
import sqlite3
from datetime import datetime

from parsing.tablelist_utils import numbers_first, make_non_cumulative
from parsing.glyph_viz import Visualizer


from ml.fontmodel import FontModel
from ml.tokenizer import Tokenizer
from api_utils.config import LOAD_MODELS, LOCAL_DEBUG, path_dtype, device, threads, FontRunStage, jwt_public_key, NUM_GLYPHS
from config import DecodeInstruction, DecodeType, SamplingType, operators

from base64 import encodebytes
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import json

import sys
sys.path.insert(0, './ml')

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

vectorization_blueprint = Blueprint('vectorization', __name__)

if LOAD_MODELS:
    font_model = torch.load('./models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', weights_only=False).to('cuda', dtype=path_dtype)
    # font_model = torch.quantization.quantize_dynamic(
    #     font_model, {torch.nn.Linear}, dtype=path_dtype
    # )
    font_model.eval()
    print("Vectorization model loaded")
vectorization_blueprint.logger.info("Vectorization model loaded")


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


class VectorizationThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        self.output = None
        self.image = None
        self.decode_instr = None
        self.log_file = None
        self.terminate_cond = threading.Event()
        self.email = None
        self.font_run_id = None
        self.character = None
        super().__init__()

    def run(self):
        im = self.image.unsqueeze(1)
        with torch.no_grad():
            sequence = font_model.decode(im, None, self.decode_instr, self.log_file, self.terminate_cond.is_set)[0].cpu().detach().numpy().flatten()
        img_arr = numeric_tokens_to_im(sequence, self.decode_instr)

        if self.terminate_cond.is_set():
            return

        toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence if tokenizer.reverse_map(tk.item(), use_int=True) not in ['<PAD2>', '<PAD>']]
            
        self.output = img_arr
        # Update database with vectorized image
        if self.email and self.font_run_id and self.character is not None:
            save_vectorized_path(self.email, self.font_run_id, self.character, toks)
            save_vectorized_image(self.email, self.font_run_id, self.character, self.output.astype(np.uint8))
        self.progress = "complete"

    def terminate(self):
        self.terminate_cond.set()



#### PATH API ####

@vectorization_blueprint.route('/<string:font_run_id>/<int:character>/sample', methods=['POST'])
@cross_origin(supports_credentials=True)
def sample_path(font_run_id, character):
    print("Received path request for font run id: ", font_run_id)
    global threads

    # Authentication check
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

    data = flask.request.get_json()
    image = data.get('image', None)
    
    if image is None:
        return make_response(jsonify({'error': 'Expected image'}), 400)
    
    if character is None or not isinstance(character, int) or character < 0 or character >= NUM_GLYPHS:
        return make_response(jsonify({'error': f'Expected character as integer between 0 and {NUM_GLYPHS-1}'}), 400)

    if LOCAL_DEBUG:
        response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/vectorization/{font_run_id}/{character}/get_progress'}))
        # response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT font_run_vectorization_complete FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id))
    font_run_vectorization_complete = cursor.fetchone()[0]
    font_run_vectorization_complete = json.loads(font_run_vectorization_complete.decode('utf-8'))
    font_run_vectorization_complete[character] = False
    cursor.execute('UPDATE font_runs SET font_run_vectorization_complete = ? WHERE (email, font_run_id) = (?, ?)', (bytes(json.dumps(font_run_vectorization_complete), 'utf-8'), email, font_run_id))
    conn.commit()
    conn.close()
    
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
            return make_response(jsonify({'error': 'Configuration error'}), 500)
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
    
    # Use font_run_id as thread key instead of just thread_id
    if font_run_id not in threads:
        threads[font_run_id] = {}
    
    threads[font_run_id][character] = VectorizationThread()
    threads[font_run_id][character].image = image
    threads[font_run_id][character].decode_instr = decode_instr
    threads[font_run_id][character].log_file = f"{character}.log"
    threads[font_run_id][character].email = email
    threads[font_run_id][character].font_run_id = font_run_id
    threads[font_run_id][character].character = character
    threads[font_run_id][character].start()

    response = make_response(jsonify({'progress': 0, 'url_extension': f'/api/vectorization/{font_run_id}/{character}/get_progress'}))
    # response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response
    
@vectorization_blueprint.route('/<string:font_run_id>/<int:character>/get_progress', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_thread_progress_path(font_run_id, character):
    global threads

    # Authentication and authorization check
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

    if LOCAL_DEBUG:
        im = (torch.zeros((128, 128)) * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
        save_vectorized_image(email, font_run_id, character, im)
        img_io = BytesIO()
        img = Image.fromarray(im.astype(np.uint8))
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        response_pre = encodebytes(img_io.getvalue()).decode('ascii')
        response = make_response(jsonify([response_pre]))
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    if font_run_id not in threads or character not in threads[font_run_id] or type(threads[font_run_id][character]) != VectorizationThread:
        return make_response(jsonify({'error': 'Thread not found'}), 404)
    
    if threads[font_run_id][character].progress == "complete":
        im = threads[font_run_id][character].output
        img_io = BytesIO()
        img = Image.fromarray(im.astype(np.uint8))
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        response_pre = encodebytes(img_io.getvalue()).decode('ascii')
        response = make_response(jsonify([response_pre]))
        # response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    else:
        # Check if there's a progress log file for this thread
        log_file_path = f"{character}.log"
        
        tok_seq = None
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as log_file:
                    tok_seq = log_file.read().strip()
            except Exception as e:
                print(f"Error reading progress log file: {e}")
        if not tok_seq or len(tok_seq.split(" ")) <= 7:
            return make_response(jsonify({'progress': threads[font_run_id][character].progress}))
        else:
            im = numeric_tokens_to_im(torch.tensor(list(map(int, tok_seq.split(" "))), dtype=torch.int32), threads[font_run_id][character].decode_instr)
            img_io = BytesIO()
            img = Image.fromarray(im.astype(np.uint8))
            img.save(img_io, format='JPEG')
            img_io.seek(0)
            response_pre = encodebytes(img_io.getvalue()).decode('ascii')
            response = make_response(jsonify({'image': [response_pre], 'progress': len(tok_seq.split(" ")) / threads[font_run_id][character].decode_instr.max_seq_len}))
            # response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response

@vectorization_blueprint.route('/<string:font_run_id>/<int:character>/cancel', methods=['POST'])
@cross_origin(supports_credentials=True)
def cancel_thread(font_run_id, character):
    global threads

    # Authentication and authorization check
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
    
    if font_run_id in threads and character in threads[font_run_id]:
        threads[font_run_id][character].terminate()
        del threads[font_run_id][character]
        # Clean up empty font_run_id entries
        if not threads[font_run_id]:
            del threads[font_run_id]
    return make_response(jsonify({'success': True}))


def save_vectorized_path(email, font_run_id, character, path):
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    cursor.execute(f'UPDATE font_runs SET font_run_vector_paths_{character} = ? WHERE (email, font_run_id) = (?, ?)', (bytes(json.dumps(path), 'utf-8'), email, font_run_id))
    conn.commit()
    conn.close()

def save_vectorized_image(email, font_run_id, character, output):
    try:
        # Convert image to base64 JPEG
        img_io = BytesIO()
        img = Image.fromarray(output.astype(np.uint8))
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        image_data = encodebytes(img_io.getvalue()).decode('ascii')
        
        # Update database
        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        cursor.execute('SELECT font_run_vectorization_complete FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id))
        font_run_vectorization_complete = cursor.fetchone()[0]
        font_run_vectorization_complete = json.loads(font_run_vectorization_complete.decode('utf-8'))
        font_run_vectorization_complete[character] = True
        cursor.execute(f'UPDATE font_runs SET (font_run_vectorized_images_{character}, font_run_vectorization_complete) = (?, ?) WHERE (email, font_run_id) = (?, ?)', 
                        (image_data, bytes(json.dumps(font_run_vectorization_complete), 'utf-8'), email, font_run_id))
        conn.commit()
        conn.close()
        print(f"Saved vectorized image for character {character} to database")
    except Exception as e:
        print(f"Error saving vectorized image to database: {e}")