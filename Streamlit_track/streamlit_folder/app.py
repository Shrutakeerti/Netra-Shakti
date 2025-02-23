from flask import Flask, request, jsonify
import sqlite3
import hashlib
import tensorflow as tf
import numpy as np
import json
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT,
                        name TEXT,
                        age INTEGER,
                        address TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    name = data.get("name")
    age = data.get("age")
    address = data.get("address")
    
    if not (username and password and name and age and address):
        return jsonify({"message": "All fields are required!"}), 400
    
    hashed_password = hash_password(password)
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, name, age, address) VALUES (?, ?, ?, ?, ?)",
                       (username, hashed_password, name, age, address))
        conn.commit()
        return jsonify({"message": "User registered successfully!"})
    except sqlite3.IntegrityError:
        return jsonify({"message": "Username already exists!"}), 400
    finally:
        conn.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = hash_password(data.get("password"))
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return jsonify({"message": "Login successful!"})
    else:
        return jsonify({"message": "Invalid credentials!"}), 401

# Load trained eye disease detection models
try:
    eye_disease_model = tf.keras.models.load_model("C:/Users/DELL/Desktop/streamlit_app/models/eye_disease_model.h5")
    inner_eye_model = tf.keras.models.load_model("C:/Users/DELL/Desktop/streamlit_app/models/inner_eyes_models.h5")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)  # Stop execution if models are not found

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # Handle error case where image is not read correctly
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"message": "No file uploaded!"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"message": "No selected file!"}), 400
    
    model_choice = request.form.get("model")
    if model_choice not in ["eye_disease", "inner_eye"]:
        return jsonify({"message": "Invalid model choice! Choose 'eye_disease' or 'inner_eye'"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    image = preprocess_image(file_path)
    if image is None:
        os.remove(file_path)  # Clean up file if corrupted
        return jsonify({"message": "Invalid image file!"}), 400
    
    model = eye_disease_model if model_choice == "eye_disease" else inner_eye_model
    prediction = model.predict(image)
    result = "Disease Detected" if prediction[0][0] > 0.5 else "No Disease"

    os.remove(file_path)  # Clean up after prediction
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
