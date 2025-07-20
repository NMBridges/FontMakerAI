import flask
from flask import make_response, jsonify, Blueprint
from flask_cors import CORS, cross_origin
import jwt
from api_utils.config import jwt_public_key, FontRunStage, NUM_GLYPHS
import sqlite3
import json
import numpy as np
import uuid
from datetime import datetime

dashboard_blueprint = Blueprint('dashboard', __name__)


@dashboard_blueprint.route('/font-runs', methods=['GET'])
@cross_origin(supports_credentials=True)
def font_runs():
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

        new_line_tab = ',\n\t'

        # get font runs for user
        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS font_runs (
                email TEXT NOT NULL,
                font_run_id BLOB NOT NULL,
                font_run_text TEXT,
                font_run_stage INTEGER DEFAULT 0,
                {new_line_tab.join([f"font_run_bitmap_images_{i} BLOB" for i in range(NUM_GLYPHS)])},
                font_run_vectorization_complete BLOB NOT NULL,
                {new_line_tab.join([f"font_run_vectorized_images_{i} BLOB" for i in range(NUM_GLYPHS)])},
                {new_line_tab.join([f"font_run_vector_paths_{i} BLOB" for i in range(NUM_GLYPHS)])},
                font_run_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                font_run_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.execute('SELECT font_run_id, font_run_text, font_run_stage, font_run_created_at, font_run_updated_at FROM font_runs WHERE email = ?', (email,))
        font_runs = [{'id': font_run[0], 'prompt': font_run[1], 'status': FontRunStage(font_run[2]).name, 'created_at': font_run[3], 'updated_at': font_run[4]} for font_run in cursor.fetchall()]
        conn.close()
        print("font_runs: ", font_runs)
        return make_response(jsonify({'success': True, 'fontRuns': font_runs}))
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)
    
@dashboard_blueprint.route('/create-font-run', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_font_run():
    auth_header = flask.request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        email = payload['email']

        # get font runs for user
        conn = sqlite3.connect('font_runs.db')
        cursor = conn.cursor()
        # create new font run
        font_run = {
            'font_run_id': str(uuid.uuid4()),
            'font_run_text': '[No description provided]',
            'font_run_stage': FontRunStage.EMPTY_DESCRIPTION.value,  # Stage 0: Empty Description
            'font_run_vectorization_complete': bytes(json.dumps([False] * NUM_GLYPHS), 'utf-8')
        }
        for i in range(NUM_GLYPHS):
            font_run[f'font_run_bitmap_images_{i}'] = memoryview(np.ones((128, 128, 3), dtype=np.uint8) * 255)
            font_run[f'font_run_vectorized_images_{i}'] = memoryview(np.zeros((128, 128, 3), dtype=np.uint8))
            font_run[f'font_run_vector_paths_{i}'] = bytes(json.dumps(['rmoveto', 0, 0, 'endchar']), 'utf-8')
        cursor.execute(f'INSERT INTO font_runs (email, font_run_id, font_run_text, font_run_stage, {", ".join([f"font_run_bitmap_images_{i}" for i in range(NUM_GLYPHS)])}, font_run_vectorization_complete, {", ".join([f"font_run_vectorized_images_{i}" for i in range(NUM_GLYPHS)])}, {", ".join([f"font_run_vector_paths_{i}" for i in range(NUM_GLYPHS)])}) VALUES (?, ?, ?, ?, {", ".join([f"?" for _ in range(3 * NUM_GLYPHS + 1)])})', (email, font_run['font_run_id'], font_run['font_run_text'], font_run['font_run_stage'], *[font_run[f'font_run_bitmap_images_{i}'] for i in range(NUM_GLYPHS)], font_run['font_run_vectorization_complete'], *[font_run[f'font_run_vectorized_images_{i}'] for i in range(NUM_GLYPHS)], *[font_run[f'font_run_vector_paths_{i}'] for i in range(NUM_GLYPHS)]))
        conn.commit()
        conn.close()
        return make_response(jsonify({'success': True, 'fontRunId': font_run['font_run_id']}))
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)