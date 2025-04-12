import os
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__, template_folder='template', static_folder='static')

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'users.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the trained model and encoder
model = joblib.load(os.path.join('aimodel', 'disease_prediction_model.pkl'))
mlb = joblib.load(os.path.join('aimodel', 'symptom_encoder.pkl'))


# Create a User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create the DB if it doesn't exist yet
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if the email and password match an existing user
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials, please try again."
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Create a new user object
        user = User(username=username, email=email, password=password)
        
        # Add the new user to the session and commit to the database
        db.session.add(user)
        db.session.commit()
        
        # Redirect to the dashboard after successful registration
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    # Query all users from the database
    users = User.query.all()
    return render_template('dashboard.html', users=users)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    symptoms = request.json.get('symptoms')
    encoded_symptoms = mlb.transform([symptoms])
    prediction = model.predict(encoded_symptoms)
    disease = prediction[0]
    return jsonify({'disease': disease})

if __name__ == '__main__':
    app.run(debug=True)
