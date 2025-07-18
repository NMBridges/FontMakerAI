import flask
import sqlite3
import bcrypt
import jwt
import json
import uuid
from datetime import datetime
from flask import make_response, jsonify, Blueprint
from flask_cors import CORS, cross_origin
from api_utils.config import jwt_public_key, jwt_private_key


### AUTH API ####

auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/signup', methods=['POST'])
def signup():
    data = flask.request.get_json()
    email = data.get('email', None)
    password = data.get('password', None)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Check if the users table exists, and create it if it doesn't
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password BLOB NOT NULL,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Check if the user already exists
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    if user:
        return make_response(jsonify({'error': 'User already exists'}), 400)
    
    # Insert the user into the database
    new_uuid = str(uuid.uuid4())
    cursor.execute('INSERT INTO users (email, password, user_id) VALUES (?, ?, ?)', (email, bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()), new_uuid))
    conn.commit()
    conn.close()
    token = jwt.encode({'email': email, 'uuid': new_uuid}, key=jwt_private_key, algorithm='RS256')
    return make_response(jsonify({'success': True, 'authToken': token, 'userId': email}))

@auth_blueprint.route('/login', methods=['POST'])
@cross_origin(supports_credentials=True)
def login():
    data = flask.request.get_json()
    email = data.get('email', None)
    password = data.get('password', None)
    print("Received login request, email: ", email, "password: ", password)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Check if the users table exists, and create it if it doesn't
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password BLOB NOT NULL,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Check if the user exists
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        token = jwt.encode({'email': email, 'uuid': user[4]}, key=jwt_private_key, algorithm='RS256')
        return make_response(jsonify({'success': True, 'authToken': token, 'userId': email}))
    else:
        return make_response(jsonify({'error': 'Invalid email or password'}), 401)
    
@auth_blueprint.route('/verify-auth-token', methods=['GET'])
def verify_auth_token():
    auth_header = flask.request.headers.get('Authorization', None)
    auth_token = None
    if auth_header and auth_header.startswith('Bearer '):
        auth_token = auth_header.split(' ', 1)[1]
    if auth_token is None:
        return make_response(jsonify({'error': 'No auth token provided'}), 400)
    
    try:
        payload = jwt.decode(auth_token, key=jwt_public_key, algorithms=['RS256'])
        return make_response(jsonify({'success': True, 'userId': payload['email']}))
    except jwt.InvalidTokenError:
        return make_response(jsonify({'error': 'Invalid auth token'}), 401)