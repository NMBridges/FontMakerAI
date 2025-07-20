from flask import Blueprint, request, jsonify, make_response, send_file
from flask_cors import cross_origin
import sqlite3
import jwt
from api_utils.config import jwt_public_key, FontRunStage, NUM_GLYPHS
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import encodebytes
import json
import os

fontrun_blueprint = Blueprint('fontrun', __name__)

@fontrun_blueprint.route('/<string:font_run_id>/updateStage', methods=['POST'])
@cross_origin(supports_credentials=True)
def update_stage(font_run_id):
    print("Updating stage for font run: ", font_run_id)

    # Authentication check
    auth_header = request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        email = payload['email']
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)
    
    # Get stage from request body
    data = request.get_json()
    if not data or 'stage' not in data:
        return make_response(jsonify({'error': 'Missing stage in request'}), 400)

    stage = data['stage']
    
    # Validate stage value
    if not isinstance(stage, int) or stage < 0 or stage > 4:
        return make_response(jsonify({'error': 'Invalid stage value. Must be integer between 0-4'}), 400)
    
    # Database operations
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    
    try:
        # Authorization check - verify user owns this font run
        cursor.execute('SELECT email FROM font_runs WHERE (font_run_id, email) = (?, ?)', (font_run_id, email))
        result = cursor.fetchone()
        
        if not result:
            return make_response(jsonify({'error': 'Font run not found'}), 404)
        
        # Update the stage
        cursor.execute(
            'UPDATE font_runs SET font_run_stage = ?, font_run_updated_at = CURRENT_TIMESTAMP WHERE (font_run_id, email) = (?, ?)',
            (FontRunStage(stage).value, font_run_id, email)
        )
        conn.commit()
        conn.close()
        
        return make_response(jsonify({'success': True, 'message': 'Stage updated successfully'}))
        
    except sqlite3.Error as e:
        conn.close()
        return make_response(jsonify({'error': f'Database error: {str(e)}'}), 500)
    
    
@fontrun_blueprint.route('/<string:font_run_id>/data', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_font_run_data(font_run_id):
    print("Fetching font run data for: ", font_run_id)
    
    # Authentication check
    auth_header = request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        email = payload['email']
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)
    
    # Database operations
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    
    try:
        # Authorization check - verify user owns this font run
        cursor.execute('SELECT email FROM font_runs WHERE (font_run_id, email) = (?, ?)', (font_run_id, email))
        result = cursor.fetchone()
        
        if not result:
            return make_response(jsonify({'error': 'Font run not found'}), 404)
        
        # Fetch basic font run data
        cursor.execute('SELECT font_run_text, font_run_stage FROM font_runs WHERE font_run_id = ?', (font_run_id,))
        basic_data = cursor.fetchone()
        
        if not basic_data:
            return make_response(jsonify({'error': 'Font run data not found'}), 404)
        
        prompt, stage = basic_data
        
        # Fetch bitmap images (generated images)
        bitmap_fields = [f'font_run_bitmap_images_{i}' for i in range(26)]
        cursor.execute(f'SELECT {", ".join(bitmap_fields)} FROM font_runs WHERE font_run_id = ?', (font_run_id,))
        bitmap_data = cursor.fetchone()
        
        # Convert bitmap images to base64 if they exist
        images = []
        if bitmap_data:
            for i in range(26):
                if bitmap_data[i] is not None:
                    try:
                        # Check if it's base64 encoded JPEG (from diffusion process)
                        if isinstance(bitmap_data[i], str):
                            images.append(bitmap_data[i])
                        else:
                            # Convert numpy array to base64 JPEG
                            img_array = np.frombuffer(bitmap_data[i], dtype=np.uint8).reshape((128, 128, 3))
                            grayscale_img = np.mean(img_array, axis=2).astype(np.uint8)
                            
                            img_io = BytesIO()
                            img = Image.fromarray(grayscale_img).convert('RGB')
                            img.save(img_io, format='JPEG')
                            img_io.seek(0)
                            images.append(encodebytes(img_io.getvalue()).decode('ascii'))
                    except Exception as e:
                        print(f"Error processing bitmap image {i}: {e}")
                        images.append(None)
                else:
                    images.append(None)
        
        # Fetch vectorized images
        vectorized_fields = [f'font_run_vectorized_images_{i}' for i in range(26)]
        cursor.execute(f'SELECT {", ".join(vectorized_fields)} FROM font_runs WHERE font_run_id = ?', (font_run_id,))
        vectorized_data = cursor.fetchone()
        
        # Convert vectorized images to base64 if they exist
        vectorized_images = []
        if vectorized_data:
            for i in range(26):
                if vectorized_data[i] is not None:
                    try:
                        # Similar processing for vectorized images
                        if isinstance(vectorized_data[i], str):
                            vectorized_images.append(vectorized_data[i])
                        else:
                            # Convert numpy array to base64 JPEG
                            img_array = np.frombuffer(vectorized_data[i], dtype=np.uint8).reshape((128, 128, 3))
                            grayscale_img = np.mean(img_array, axis=2).astype(np.uint8)
                            
                            img_io = BytesIO()
                            img = Image.fromarray(grayscale_img).convert('RGB')
                            img.save(img_io, format='JPEG')
                            img_io.seek(0)
                            vectorized_images.append(encodebytes(img_io.getvalue()).decode('ascii'))
                    except Exception as e:
                        print(f"Error processing vectorized image {i}: {e}")
                        vectorized_images.append(None)
                else:
                    vectorized_images.append(None)

        vectorization_complete = cursor.execute('SELECT font_run_vectorization_complete FROM font_runs WHERE (email, font_run_id) = (?, ?)', (email, font_run_id)).fetchone()[0]
        vectorization_complete = json.loads(vectorization_complete.decode('utf-8'))

        # Prepare response data
        response_data = {
            'prompt': prompt or '',
            'stage': stage,
            'images': images,
            'vectorizedImages': vectorized_images,
            'vectorizationComplete': vectorization_complete,
            'fontFileUrl': None  # TODO: Add font file URL when implemented
        }
        
        return make_response(jsonify({'success': True, 'data': response_data}))
        
    except sqlite3.Error as e:
        return make_response(jsonify({'error': f'Database error: {str(e)}'}), 500)
    finally:
        conn.close()
    
@fontrun_blueprint.route('/<string:font_run_id>/download', methods=['GET'])
@cross_origin(supports_credentials=True)
def download_font_file(font_run_id):
    # authn and authz
    auth_header = request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        email = payload['email']

    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)
    
    # get font file from db
    conn = sqlite3.connect('font_runs.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT font_run_text, {", ".join([f"font_run_vector_paths_{i}" for i in range(NUM_GLYPHS)])} FROM font_runs WHERE (font_run_id, email) = (?, ?)', (font_run_id, email))
    font_run = cursor.fetchone()
    prompt = font_run[0]
    vector_paths = font_run[1:]
    conn.close()

    if not prompt or not vector_paths:
        return make_response(jsonify({'error': 'Prompt or vector paths not found'}), 404)
    
    tablelists = {}
    for i in range(NUM_GLYPHS):
        tablelists[f'{chr(65 + i)}'] = json.loads(vector_paths[i].decode('utf-8'))
    
    family_name = prompt.replace(" ", "-")
    # create font file
    try:
        create_font(family_name, tablelists)
        
        return send_file(f'./user_fonts/{family_name}.otf', as_attachment=True)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error': f'Error creating font file: {str(e)}'}), 500)
    finally:
        os.remove(f'./user_fonts/{family_name}.otf')



### FONT CREATION


from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.fontBuilder import FontBuilder

class Tokenizer:
    def __init__(self, min_number : int = -1500, max_number : int = 1500, possible_operators : list[str] = [],
                sos_token : str = '<SOS>', eos_token : str = '<EOS>', pad_token : str = '<PAD>'):

        self.min_number = min_number
        self.max_number = max_number
        self.possible_operators = possible_operators
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.pad2_token = '<PAD2>'

        self.map = {
            pad_token: 0,
            sos_token: 1,
            eos_token: 2,
            self.pad2_token: 3,
        }
        self.special_tokens_len = len(self.map)
        self.num_tokens = self.special_tokens_len + len(possible_operators) + max_number - min_number + 1

        run_val = self.special_tokens_len

        for operator in possible_operators:
            if operator in self.map:
                raise Exception(f"Cannot name operator {operator}; name already used")
            self.map[operator] = run_val
            run_val += 1

        for number in range(min_number, max_number + 1):
            self.map[f'{number}'] = run_val
            run_val += 1

    def __getitem__(self, key : str) -> int:
        return self.map[key]

    def reverse_map(self, index : int, use_int : bool = False) -> str:
        if index < 0 or index >= self.num_tokens:
            raise Exception(f"Invalid index. Index must be between {0} and {self.num_tokens-1} (inclusive")
        elif index < self.special_tokens_len:
            return [self.pad_token, self.sos_token, self.eos_token, self.pad2_token][index]
        elif index < self.special_tokens_len + len(self.possible_operators):
            return self.possible_operators[index - self.special_tokens_len]
        elif use_int:
            return self.min_number + index - (self.special_tokens_len + len(self.possible_operators))
        else:
            return f'{self.min_number + index - (self.special_tokens_len + len(self.possible_operators))}'

operators = [
    "rmoveto", # 4
    "hmoveto",
    "vmoveto",
    "rlineto", # 7
    "hlineto",
    "vlineto",
    "rrcurveto", # 10
    "hhcurveto",
    "vvcurveto",
    "hvcurveto",
    "vhcurveto",
    "rcurveline",
    "rlinecurve",
    "flex",
    "hflex",
    "hflex1",
    "flex1",
    "hstem",
    "vstem",
    "hstemhm",
    "vstemhm",
    "hintmask",
    "cntrmask",
    "callsubr",
    "callgsubr",
    "vsindex",
    "blend",
    "endchar", # 31
]
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
cumulative = True
vocab_size = tokenizer.num_tokens

def operator_first(tablelist : list) -> list:
    '''
    Reorders a tablelist such that the operators are before their numeric arguments.

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to reorder

    Returns:
    --------
    list: the reordered tablelist
    '''
    out_list = [0] * len(tablelist)
    op_idx = 0
    for col in range(len(tablelist)):
        if tablelist[col] not in tokenizer.possible_operators:
            out_list[col + 1] = tablelist[col]
        else:
            out_list[op_idx] = tablelist[col]
            op_idx = col + 1
    return out_list


def draw_tablelist(pen, tablelist):
    # tablelist = operator_first(num_first_tablelist)
    i = 0
    while i < len(tablelist):
        if tablelist[i] == 'rmoveto':
            pen.moveTo((tablelist[i+1], tablelist[i+2]))
            i += 3
        elif tablelist[i] == 'rlineto':
            pen.lineTo((tablelist[i+1], tablelist[i+2]))
            i += 3
        elif tablelist[i] == 'rrcurveto':
            pen.curveTo((tablelist[i+1], tablelist[i+2]), (tablelist[i+3], tablelist[i+4]), (tablelist[i+5], tablelist[i+6]))
            i += 7
        elif tablelist[i] == 'endchar':
            pen.closePath()
            return
        else:
            raise ValueError(f"Unknown operator: {tablelist[i]}")


def create_font(family_name : str, tablelists : dict[str, list[int | str]]) -> None:
    font_builder = FontBuilder(1024, isTTF=False)
    
    styleName = "Normal"
    version = "0.1"

    nameStrings = dict(
        familyName=dict(en=family_name),
        styleName=dict(en=styleName),
        uniqueFontIdentifier="fontBuilder: " + family_name + "." + styleName,
        fullName=family_name + "-" + styleName,
        psName=family_name + "-" + styleName,
        version="Version " + version,
    )

    tablelists[".notdef"] = ['rmoveto', 0, 0, 'endchar']

    glyphs = {}
    base_width = 300

    for glyph_name, tablelist in tablelists.items():
        pen = T2CharStringPen(base_width, None)
        draw_tablelist(pen, tablelist)
        glyphs[glyph_name] = pen.getCharString()

    metrics = {}
    advanceWidths = {".notdef": 0, **{k: base_width for k in tablelists.keys() if k != ".notdef"}}
    font_builder.setupGlyphOrder([".notdef", *[k for k in tablelists.keys() if k != ".notdef"]])
    font_builder.setupCFF(nameStrings["psName"], {"FullName": nameStrings["psName"]}, glyphs, {})
    lsb = {gn: cs.calcBounds(None)[0] for gn, cs in glyphs.items()}
    for gn, advanceWidth in advanceWidths.items():
        metrics[gn] = (advanceWidth, lsb[gn])
    font_builder.setupCharacterMap({ord(k): k for k, _ in tablelists.items() if k != ".notdef"})
    font_builder.setupHorizontalMetrics(metrics)
    font_builder.setupHorizontalHeader(ascent=824, descent=-200)
    font_builder.setupNameTable(nameStrings)
    font_builder.setupOS2(sTypoAscender=824, usWinAscent=824, usWinDescent=200)
    font_builder.setupPost()
    if not os.path.exists('./user_fonts'):
        os.makedirs('./user_fonts')
    font_builder.save(f'./user_fonts/{family_name}.otf')