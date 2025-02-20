from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image

 
app = Flask(__name__)
app.secret_key = "f1a4fd45237702d7e84bcae2594f6a4d"

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
 

# Create the database
with app.app_context():
    db.create_all()




# Load ML models for inner and outer eye diseases
inner_eye_model = tf.keras.models.load_model(r"D:\Diversion_2k25\eye_analysis\models\inner_eyes_models.h5")
outer_eye_model = tf.keras.models.load_model(r"D:\Diversion_2k25\eye_analysis\models\eye_disease_model.h5")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def home():
    return render_template('index.html')



#Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        #check if email already exists
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "Email already registered!!"
        
        # Store user in DB
        new_user = User(name=name, email=email, age=age, gender=gender)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for("store"))
    
    return render_template("register.html")


# Store Page - Display All Users
@app.route("/store")
def store():
    users = User.query.all()
    return render_template("store.html", users=users)



# Login Page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'password':  # Basic authentication
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            flash("Invalid username or password. Try again!", "error")
    
    return render_template('login.html')

#upload.html page 

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        inner_eye_img = request.files.get('inner_eye')
        outer_eye_img = request.files.get('outer_eye')

        predictions = {}

        # Process inner eye image
        if inner_eye_img and allowed_file(inner_eye_img.filename):
            filename = secure_filename(inner_eye_img.filename)
            inner_eye_path = os.path.join("uploads", filename)
            inner_eye_img.save(inner_eye_path)
            img_array = preprocess_image(inner_eye_path)
            inner_pred = inner_eye_model.predict(img_array)
            predictions["inner_eye_disease"] = f"Inner Eye Disease: {np.argmax(inner_pred)}"

        # Process outer eye image
        if outer_eye_img and allowed_file(outer_eye_img.filename):
            filename = secure_filename(outer_eye_img.filename)
            outer_eye_path = os.path.join("uploads", filename)
            outer_eye_img.save(outer_eye_path)
            img_array = preprocess_image(outer_eye_path)
            outer_pred = outer_eye_model.predict(img_array)
            predictions["outer_eye_disease"] = f"Outer Eye Disease: {np.argmax(outer_pred)}"

        
            print(predictions)
        return render_template("result.html", predictions=predictions)

    return render_template('upload.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
