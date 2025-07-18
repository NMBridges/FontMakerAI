from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin
import sqlite3
import jwt
from api_utils.config import jwt_public_key, FontRunStage
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import encodebytes
import json

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
    
    
