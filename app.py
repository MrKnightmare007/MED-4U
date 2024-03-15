from flask import Flask, render_template,url_for, request, Response, jsonify, session, flash,redirect
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
from flask import *
import sqlite3, hashlib, os
from werkzeug.utils import secure_filename
from instamojo_wrapper import Instamojo
import requests
from web3 import Web3
import time, json, os
from web3.middleware import geth_poa_middleware
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import io
from PIL import Image
import replicate
import google.generativeai as genai
import cv2
from keras.models import load_model
import random
from threading import Thread
from os import environ as env
from urllib.parse import quote_plus, urlencode
import uuid
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")
oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

# Load the saved model
model_filename = 'mediplus-lite/Diabetes.joblib'  # Or .sav if you used pickle
loaded_model = joblib.load(model_filename)

# Function to get user input from the HTML form
def get_user_input(request):
    parameters = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    user_data = {}

    for param in parameters:
        value = request.form.get(param)
        user_data[param] = float(value)

    return pd.DataFrame(user_data, index=[0])

# Function to generate preventive measures and dietary plan
def generate_recommendations():
    preventive_measures = [
        "1. **Manage Blood Sugar:** Regularly monitor blood sugar levels and adjust medication or lifestyle as needed",
        "2. **Healthy Diet:** Follow a balanced diet with plenty of fruits, vegetables, whole grains, and lean protein.",
        "3. **Regular Exercise:** Engage in moderate-intensity physical activity for at least 150 minutes per week.",
        "4. **Weight Management:** Aim for a healthy weight and avoid becoming overweight or obese.",
        "5. **Blood Pressure Control:** Keep blood pressure below 140/90 mmHg through lifestyle changes or medication.",
        "6. **Smoking Cessation:** Quit smoking, as it increases the risk of diabetes complications.",
        "7. **Regular Check-ups:** Schedule regular appointments with your doctor for comprehensive health evaluations."
    ]

    dietary_plan = [
        "1. **Carbohydrates:** Choose whole grains, fruits, and vegetables that are high in fiber and low in sugar.",
        "2. **Protein:** Include lean protein sources such as fish, chicken, beans, and tofu in your diet.",
        "3. **Fats:** Limit unhealthy saturated and trans fats, and opt for healthy fats from olive oil, nuts, and seeds.",
        "4. **Sugar:** Avoid sugary drinks, processed foods, and sweets.",
        "5. **Serving Sizes:** Be mindful of portion sizes to control calorie intake.",
        "6. **Meal Timing:** Eat regular meals and snacks throughout the day to help keep blood sugar levels stable.",
        "7. **Monitor Blood Sugar:** Regularly check blood sugar levels before and after meals to assess the impact of your diet.",
    ]

    return preventive_measures, dietary_plan

# Load the trained model
with open('mediplus-lite/liver.joblib', 'rb') as model_file:
    best_classifier = joblib.load(model_file)

# Define the column names
column_names = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

# Mapping of user-friendly column names to the actual column names
column_name_mapping = {
    'Age': 'Age',
    'Gender': 'Gender',
    'Total_Bilirubin': 'Total Bilirubin',
    'Direct_Bilirubin': 'Direct Bilirubin',
    'Alkaline_Phosphotase': 'Alkaline Phosphotase',
    'Alamine_Aminotransferase': 'Alamine Aminotransferase',
    'Aspartate_Aminotransferase': 'Aspartate Aminotransferase',
    'Total_Protiens': 'Total Proteins',
    'Albumin': 'Albumin',
    'Albumin_and_Globulin_Ratio': 'Albumin and Globulin Ratio'
}

# Load the trained model
model_filename = 'kidney/random_forest_model.joblib'  # Update with the correct path
rd_clf = joblib.load(model_filename)

app.secret_key = 'random string'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = "f684818e6adcfd2f34c8b503fb65c0d9"  # Change this to a random secret key

os.environ['GOOGLE_API_KEY'] = "AIzaSyDvIm1BXLGilc89Knx0VWTfRhB5pFxgRYY"  # Replace with your actual Gemini API Key
os.environ["REPLICATE_API_TOKEN"] = "r8_KlXXLXKsmAs2YhezhPe8yYy7aQ5LpTD1kE2qN"
model_name = "meta/llama-2-70b-chat"

# Configure session to use signed cookies
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'emotion-chatbot'
app.config['SESSION_FILE_THRESHOLD'] = 500  # Increase if needed
#session.init_app(app)

# Global variables to store the detected emotion and conversation history
emotion_detected_key = 'emotion_detected'
conversation_history_key = 'conversation_history'

def get_system_prompt(emotion):
    return f"You are an emotion-powered chatbot. Your responses are influenced by the user's emotions. Currently, the user is {emotion}. You must act as a psychiatric counselor and professional psychologist and provide mental support to the user and also provide enough exclamations."

def detect_emotion(image_array):
    # Convert image array to bytes
    img_bytes = image_array.tobytes()

    # Create PIL Image from bytes with explicit format (JPEG)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    response = vision_model.generate_content(["If it is a human being, only say the emotion they are expressing, and if it is not a human being, just say that it is not a human being.", img])
    description = response.text
    if "neutral" in description:
        emotion = "neutral"
    else:
        emotion = description.split()[-1].strip('.')
    return emotion

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Load the trained model, scaler, and feature columns
model_rf, scaler, columns = joblib.load('mediplus-lite/gen_medicine.joblib')

# Function to make predictions using the trained model
def make_prediction(symptoms):
    try:
        # Create a DataFrame with user input symptoms
        user_input = pd.DataFrame(0, index=[0], columns=columns)
        user_input[symptoms] = 1

        # Feature scaling for user input
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model_rf.predict(user_input_scaled)

        return prediction[0]
    except Exception as e:
        print("An error occurred during prediction:", str(e))
        return None

# Load the saved model
model = load_model('mediplus-lite/my_model.h5')
# Define a mapping between class labels and disease names
class_to_disease = {
    0: "Normal",
    1: "Glaucoma",
    2: "Cataract",
    3: "Retina Disease"
}

# Function to preprocess input image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (128, 128))
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)
    # Perform inference
    predictions = model.predict(preprocessed_image)
    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    # Get the corresponding disease name from the mapping
    disease_name = class_to_disease[predicted_class]
    return disease_name

# Load the trained model
model = load_model("mediplus-lite/dental_model9.h5")

# Define a dictionary mapping class indices to class labels
class_labels = {
    0: "Calculus",
    1: "Data caries",
    2: "Gingivitis",
    3: "Hypodontia",
    4: "Mouth Ulcer",
    5: "Tooth Discoloration"
}

# Function to predict disease from an uploaded image
def predict_disease(image):
    try:
        # Resize the image to match the expected input shape of the model
        resized_image = cv2.resize(image, (256, 256))

        # Convert the image to the format expected by the model (e.g., converting to float and normalizing)
        input_image = resized_image.astype('float32') / 255.0

        # Make prediction
        predictions = model.predict(np.expand_dims(input_image, axis=0))
        predicted_class = np.argmax(predictions)

        # Get the predicted class label
        predicted_disease = class_labels[predicted_class]

        return predicted_disease
    except Exception as e:
        return str(e)

def generate_hospital_data(num_hospitals):
    data = {'Hospital_ID': [f'H{i}' for i in range(1, num_hospitals + 1)],
            'Total_Beds': [random.randint(250, 400) for _ in range(num_hospitals)],
            'ICU_Beds': [],
            'available_ICU_beds': [],
            'General_Beds': [],
            'available_General_beds': [],
            'Cardiology_Beds': [],
            'available_Cardiology_beds': [],
            'Orthopaedic_Beds': [],
            'available_Orthopaedic_beds': [],
            'Special_Beds': [],
            'available_Special_beds': [],
            'Maternity_Beds': [],
            'available_Maternity_beds': [],
            'occupied_beds': [],
            'available_beds': [],
            'ICU_Severity': [],
            'ICU_Admission_Type': [],
            'General_Severity': [],
            'General_Admission_Type': [],
            'Cardiology_Severity': [],
            'Cardiology_Admission_Type': [],
            'Orthopaedic_Severity': [],
            'Orthopaedic_Admission_Type': [],
            'Special_Severity': [],
            'Special_Admission_Type': [],
            'Maternity_Severity': [],
            'Maternity_Admission_Type': []}

    for i in range(num_hospitals):
        total_beds = data['Total_Beds'][i]

        data['ICU_Beds'].append(random.randint(10, total_beds // 4))
        data['available_ICU_beds'].append(random.randint(1, data['ICU_Beds'][i]))

        data['General_Beds'].append(random.randint(20, total_beds // 2))
        data['available_General_beds'].append(random.randint(1, data['General_Beds'][i]))

        data['Cardiology_Beds'].append(random.randint(5, total_beds // 8))
        data['available_Cardiology_beds'].append(random.randint(1, data['Cardiology_Beds'][i]))

        data['Orthopaedic_Beds'].append(random.randint(5, total_beds // 8))
        data['available_Orthopaedic_beds'].append(random.randint(1, data['Orthopaedic_Beds'][i]))

        data['Special_Beds'].append(random.randint(5, total_beds // 8))
        data['available_Special_beds'].append(random.randint(1, data['Special_Beds'][i]))

        data['Maternity_Beds'].append(random.randint(5, total_beds // 8))
        data['available_Maternity_beds'].append(random.randint(1, data['Maternity_Beds'][i]))

        data['occupied_beds'].append(random.randint(1, total_beds // 4))
        data['available_beds'].append(random.randint(1, total_beds // 4))

        for bed_type in ['ICU', 'General', 'Cardiology', 'Orthopaedic', 'Special', 'Maternity']:
            data[f'{bed_type}_Severity'].append(random.choice(['Critical', 'Moderate', 'Stable']))
            data[f'{bed_type}_Admission_Type'].append(random.choice(['Emergency', 'Scheduled']))

    df = pd.DataFrame(data)
    return df

def randomize_bed_availability(current_available, total_beds):
    change = random.randint(-2, 2)
    new_available = max(0, min(current_available + change, total_beds))
    return new_available

def update_data():
    global data
    while True:
        for index, row in data.iterrows():
            for bed_type in ['ICU', 'General', 'Cardiology', 'Orthopaedic', 'Special', 'Maternity']:
                available_bed_col = f'available_{bed_type}_beds'
                data.loc[index, available_bed_col] = randomize_bed_availability(
                    row[available_bed_col], row[f'{bed_type}_Beds']
                )

            all_specialty_beds = [f'available_{bed_type}_beds' for bed_type in 
                                ['ICU', 'General', 'Cardiology', 'Orthopaedic', 'Special', 'Maternity']]
            data.loc[index, 'available_beds'] = data.iloc[index][all_specialty_beds].sum()
            data.loc[index, 'occupied_beds'] = row['Total_Beds'] - row['available_beds']
            
        time.sleep(120)

# Initial data generation
data = generate_hospital_data(num_hospitals=10)

# Start a thread for data update
update_thread = Thread(target=update_data)
update_thread.start()

# Controllers API
@app.route("/")
def home1():
    return render_template(
        "home1.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4),
    )

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/index")

@app.route("/login1")
def login1():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )

@app.route("/logout1")
def logout1():
    session.clear()
    return redirect(
        "https://"
        + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )

appointments = {}

# Sample data for doctor types and locations
doctor_types = ["Dentist", "Cardiologist", "Dermatologist", "Orthopedic"]
locations = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"]

@app.route('/index')
def index():
    return render_template('index.html')
# Define the route for the home page
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Get input from the user
        user_input_df = get_user_input(request)

        # Make a prediction
        prediction = loaded_model.predict(user_input_df)[0]

        if prediction == 0:
            result_message = "Based on the provided input, there is no risk of diabetes."
            preventive_measures, dietary_plan = [], []  # No recommendations for low risk
        else:
            result_message = "Based on the provided input, there is a risk of diabetes."
            preventive_measures, dietary_plan = generate_recommendations()

        return render_template('diabetes.html', result=result_message, preventive_measures=preventive_measures, dietary_plan=dietary_plan)
    
    # If it's a GET request, render the form
    return render_template('diabetes.html', result=None, preventive_measures=None, dietary_plan=None)

# Function to render the index page
@app.route('/liver')
def liver():
    return render_template('liver.html', column_name_mapping=column_name_mapping)

# Function to make a prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input_data = {}
        for col in column_names:
            value = request.form[col]
            user_input_data[col] = float(value) if col != 'Gender' else value.capitalize()

        # Convert 'Gender' to numerical values using label encoding
        le = LabelEncoder()
        user_input_data['Gender'] = le.fit_transform([user_input_data['Gender']])[0]

        # Create a DataFrame with user input
        user_input_df = pd.DataFrame([user_input_data])

        # Ensure the user input DataFrame has the same columns as the training data
        user_input_df = user_input_df.reindex(columns=column_names, fill_value=0)

        # Make prediction using the trained model
        prediction = best_classifier.predict(user_input_df)[0]

        # Print the appropriate message based on the prediction
        result_message = "You have a liver disease." if prediction == 1 else "You don't have a liver disease."

        return render_template('liver.html', prediction=result_message, column_name_mapping=column_name_mapping)
    except Exception as e:
        return render_template('liver.html', error_message=str(e), column_name_mapping=column_name_mapping)

df = pd.read_csv('heart/heart.csv')
# Implementing one-hot encoding on the specified categorical features
df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], drop_first=True)

# Convert the rest of the categorical variables that don't need one-hot encoding to integer data type
features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
for feature in features_to_convert:
    df_encoded[feature] = df_encoded[feature].astype(int)

# Load the trained model
with open('heart/gradient_boosting_model.joblib', 'rb') as model_file:
    best_gb = joblib.load(model_file)

# Define the features (X) and the output labels (y)
X = df_encoded.drop('target', axis=1)

# Mapping of user-friendly column names to the actual column names
column_name_mapping = {
    'age': 'Age',
    'sex': 'Gender (0 = male, 1 = female)',
    'cp': 'Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar level (1 = above 120 mg/dl, 0 = below 120 mg/dl)',
    'restecg': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved during a stress test',
    'exang': 'Exercise-induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
    'thal': 'Thalium stress test result (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described)',
}

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')
# Function to make a prediction based on user input
@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        # Get user input from the form
        user_input = [float(request.form[col]) for col in request.form]

        # Create a NumPy array with user input
        user_input_array = np.array(user_input).reshape(1, -1)

        # Make prediction using the trained model
        prediction = rd_clf.predict(user_input_array)

        # Interpret the prediction
        result = "You have a Kidney Disease" if prediction[0] == 0 else "You have no Kidney Disease"
        
        return render_template('kidney.html', prediction=result)
    except Exception as e:
        return render_template('kideny.html', error_message=str(e))

# Function to render the index page
@app.route('/heart')
def heart():
    return render_template('heart.html', column_name_mapping=column_name_mapping)

# Function to make a prediction based on user input
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        user_input_data = {}
        for col, description in column_name_mapping.items():
            value = request.form[col]
            user_input_data[col] = float(value) if col != 'sex' else int(value)

        user_input_df = pd.DataFrame([user_input_data])
        user_input_df = pd.get_dummies(user_input_df, columns=['cp', 'restecg', 'thal'], drop_first=True)
        user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)
        prediction = best_gb.predict(user_input_df)[0]

        result_message = "Heart Disease" if prediction == 1 else "No Heart Disease"

        return render_template('heart.html', prediction=result_message, column_name_mapping=column_name_mapping)
    except Exception as e:
        return render_template('heart.html', error_message=str(e), column_name_mapping=column_name_mapping)
    
#Home page
@app.route("/home")
def home():
    loggedIn, firstName, noOfItems = getLoginDetails()
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT productId, name, price, description, image, stock FROM products')
        itemData = cur.fetchall()
        cur.execute('SELECT categoryId, name FROM categories')
        categoryData = cur.fetchall()
    itemData = parse(itemData)   
    return render_template('home.html', itemData=itemData, loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems, categoryData=categoryData)

#Fetch user details if logged in
def getLoginDetails():
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        if 'email' not in session:
            loggedIn = False
            firstName = ''
            noOfItems = 0
        else:
            loggedIn = True
            cur.execute("SELECT userId, firstName FROM users WHERE email = '" + session['email'] + "'")
            userId, firstName = cur.fetchone()
            cur.execute("SELECT count(productId) FROM kart WHERE userId = " + str(userId))
            noOfItems = cur.fetchone()[0]
    conn.close()
    return (loggedIn, firstName, noOfItems)

#Add item to cart
@app.route("/addItem", methods=["GET", "POST"])
def addItem():
    if request.method == "POST":
        name = request.form['name']
        price = float(request.form['price'])
        description = request.form['description']
        stock = int(request.form['stock'])
        categoryId = int(request.form['category'])

        #Upload image
        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imagename = filename
        with sqlite3.connect('mediplus-lite/database.db') as conn:
            try:
                cur = conn.cursor()
                cur.execute('''INSERT INTO products (name, price, description, image, stock, categoryId) VALUES (?, ?, ?, ?, ?, ?)''', (name, price, description, imagename, stock, categoryId))
                conn.commit()
                msg="Added successfully"
            except:
                msg="Error occured"
                conn.rollback()
        conn.close()
        print(msg)
        return redirect(url_for('home'))

#Remove item from cart
@app.route("/removeItem")
def removeItem():
    productId = request.args.get('productId')
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        try:
            cur = conn.cursor()
            cur.execute('DELETE FROM products WHERE productID = ' + productId)
            conn.commit()
            msg = "Deleted successsfully"
        except:
            conn.rollback()
            msg = "Error occured"
    conn.close()
    print(msg)
    return redirect(url_for('home'))

#Display all items of a category
@app.route("/displayCategory")
def displayCategory():
        loggedIn, firstName, noOfItems = getLoginDetails()
        categoryId = request.args.get("categoryId")
        with sqlite3.connect('mediplus-lite/database.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT products.productId, products.name, products.price, products.image, categories.name FROM products, categories WHERE products.categoryId = categories.categoryId AND categories.categoryId = " + categoryId)
            data = cur.fetchall()
        conn.close()
        categoryName = data[0][4]
        data = parse(data)
        return render_template('displayCategory.html', data=data, loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems, categoryName=categoryName)

@app.route("/account/profile")
def profileHome():
    if 'email' not in session:
        return redirect(url_for('home'))
    loggedIn, firstName, noOfItems = getLoginDetails()
    return render_template("profileHome.html", loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems)

@app.route("/account/profile/edit")
def editProfile():
    if 'email' not in session:
        return redirect(url_for('home'))
    loggedIn, firstName, noOfItems = getLoginDetails()
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT userId, email, firstName, lastName, address1, address2, zipcode, city, state, country, phone FROM users WHERE email = '" + session['email'] + "'")
        profileData = cur.fetchone()
    conn.close()
    return render_template("editProfile.html", profileData=profileData, loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems)

@app.route("/account/profile/changePassword", methods=["GET", "POST"])
def changePassword():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    if request.method == "POST":
        oldPassword = request.form['oldpassword']
        oldPassword = hashlib.md5(oldPassword.encode()).hexdigest()
        newPassword = request.form['newpassword']
        newPassword = hashlib.md5(newPassword.encode()).hexdigest()
        with sqlite3.connect('mediplus-lite/database.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT userId, password FROM users WHERE email = '" + session['email'] + "'")
            userId, password = cur.fetchone()
            if (password == oldPassword):
                try:
                    cur.execute("UPDATE users SET password = ? WHERE userId = ?", (newPassword, userId))
                    conn.commit()
                    msg="Changed successfully"
                except:
                    conn.rollback()
                    msg = "Failed"
                return render_template("changePassword.html", msg=msg)
            else:
                msg = "Wrong password"
        conn.close()
        return render_template("changePassword.html", msg=msg)
    else:
        return render_template("changePassword.html")

@app.route("/updateProfile", methods=["GET", "POST"])
def updateProfile():
    if request.method == 'POST':
        email = request.form['email']
        firstName = request.form['firstName']
        lastName = request.form['lastName']
        address1 = request.form['address1']
        address2 = request.form['address2']
        zipcode = request.form['zipcode']
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        phone = request.form['phone']
        with sqlite3.connect('mediplus-lite/database.db') as con:
                try:
                    cur = con.cursor()
                    cur.execute('UPDATE users SET firstName = ?, lastName = ?, address1 = ?, address2 = ?, zipcode = ?, city = ?, state = ?, country = ?, phone = ? WHERE email = ?', (firstName, lastName, address1, address2, zipcode, city, state, country, phone, email))

                    con.commit()
                    msg = "Saved Successfully"
                except:
                    con.rollback()
                    msg = "Error occured"
        con.close()
        return redirect(url_for('editProfile'))

@app.route("/loginForm")
def loginForm():
    if 'email' in session:
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error='')

@app.route("/login", methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if is_valid(email, password):
            session['email'] = email
            return redirect(url_for('home'))
        else:
            error = 'Invalid UserId / Password'
            return render_template('login.html', error=error)

@app.route("/productDescription")
def productDescription():
    loggedIn, firstName, noOfItems = getLoginDetails()
    productId = request.args.get('productId')
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT productId, name, price, description, image, stock FROM products WHERE productId = ' + productId)
        productData = cur.fetchone()
    conn.close()
    return render_template("productDescription.html", data=productData, loggedIn = loggedIn, firstName = firstName, noOfItems = noOfItems)

@app.route("/addToCart")
def addToCart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    else:
        productId = int(request.args.get('productId'))
        with sqlite3.connect('mediplus-lite/database.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT userId FROM users WHERE email = '" + session['email'] + "'")
            userId = cur.fetchone()[0]
            try:
                cur.execute("INSERT INTO kart (userId, productId) VALUES (?, ?)", (userId, productId))
                conn.commit()
                msg = "Added successfully"
            except:
                conn.rollback()
                msg = "Error occured"
        conn.close()
        return redirect(url_for('home'))

@app.route("/cart")
def cart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    loggedIn, firstName, noOfItems = getLoginDetails()
    email = session['email']
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT userId FROM users WHERE email = '" + email + "'")
        userId = cur.fetchone()[0]
        cur.execute("SELECT products.productId, products.name, products.price, products.image FROM products, kart WHERE products.productId = kart.productId AND kart.userId = " + str(userId))
        products = cur.fetchall()
    totalPrice = 0
    for row in products:
        totalPrice += row[2]
    return render_template("cart.html", products = products, totalPrice=totalPrice, loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems)

@app.route("/checkout")
def checkout():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    loggedIn, firstName, noOfItems = getLoginDetails()
    email = session['email']
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT userId FROM users WHERE email = '" + email + "'")
        userId = cur.fetchone()[0]
        cur.execute("SELECT products.productId, products.name, products.price, products.image FROM products, kart WHERE products.productId = kart.productId AND kart.userId = " + str(userId))
        products = cur.fetchall()
    totalPrice = 0
    for row in products:
        totalPrice += row[2]
    return render_template("checkout.html", products = products, totalPrice=totalPrice, loggedIn=loggedIn, firstName=firstName, noOfItems=noOfItems)

@app.route("/instamojo")
def instamojo():
    return render_template("instamojo.html")

@app.route("/removeFromCart")
def removeFromCart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    email = session['email']
    productId = int(request.args.get('productId'))
    with sqlite3.connect('mediplus-lite/database.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT userId FROM users WHERE email = '" + email + "'")
        userId = cur.fetchone()[0]
        try:
            cur.execute("DELETE FROM kart WHERE userId = " + str(userId) + " AND productId = " + str(productId))
            conn.commit()
            msg = "removed successfully"
        except:
            conn.rollback()
            msg = "error occured"
    conn.close()
    return redirect(url_for('home'))

@app.route("/logout")
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

def is_valid(email, password):
    con = sqlite3.connect('mediplus-lite/database.db')
    cur = con.cursor()
    cur.execute('SELECT email, password FROM users')
    data = cur.fetchall()
    for row in data:
        if row[0] == email and row[1] == hashlib.md5(password.encode()).hexdigest():
            return True
    return False

@app.route("/register", methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        #Parse form data    
        password = request.form['password']
        email = request.form['email']
        firstName = request.form['firstName']
        lastName = request.form['lastName']
        address1 = request.form['address1']
        address2 = request.form['address2']
        zipcode = request.form['zipcode']
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        phone = request.form['phone']

        with sqlite3.connect('mediplus-lite/database.db') as con:
            try:
                cur = con.cursor()
                cur.execute('INSERT INTO users (password, email, firstName, lastName, address1, address2, zipcode, city, state, country, phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (hashlib.md5(password.encode()).hexdigest(), email, firstName, lastName, address1, address2, zipcode, city, state, country, phone))

                con.commit()

                msg = "Registered Successfully"
            except:
                con.rollback()
                msg = "Error occured"
        con.close()
        return render_template("login.html", error=msg)

@app.route("/registerationForm")
def registrationForm():
    return render_template("register.html")

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def parse(data):
    ans = []
    i = 0
    while i < len(data):
        curr = []
        for j in range(7):
            if i >= len(data):
                break
            curr.append(data[i])
            i += 1
        ans.append(curr)
    return ans

load_dotenv()

ADMIN_ADDRESS = os.getenv("ADMIN_ADDRESS")
W3_PROVIDER = os.getenv("W3_PROVIDER")
SECRET_KEY = os.getenv("SECRET_KEY")
gasPrice = os.getenv("gasPrice")
gasLimit = os.getenv("gasLimit")

# Dictionary to store payment information with timestamps
payment_data = json.loads(open('mediplus-lite/payment_info.json', 'r').read())
app.config['SECRET_KEY'] = SECRET_KEY
w3 = Web3(Web3.HTTPProvider(W3_PROVIDER))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

scheduler = BackgroundScheduler()

def save_payment_info_to_json(payment_data):
    with open('mediplus-lite/payment_info.json', 'r+') as file:
        file_data = json.load(file)
        file_data.update(payment_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

def timestamp_to_datetime(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

app.jinja_env.globals.update(timestamp_to_datetime=timestamp_to_datetime)

def check_payments():
    for payment_address, data in payment_data.items():
        timestamp = data['timestamp']
        current_time = time.time()

        if current_time - timestamp <= 900:
            balance = w3.eth.get_balance(payment_address)

            if balance > 0:
                try:
                    print(f"Payment received for {w3.from_wei(balance, 'ether')}")
                    send_payment_info_to_admin(payment_address, payment_data[payment_address]['private_key'], balance)
                except:
                    pass

scheduler.add_job(check_payments, 'interval', seconds=10)
scheduler.start()

@app.route('/payment')
def payment():
    return render_template('payment.html')

@app.route('/generate_payment_address', methods=['POST'])
def generate_payment_address():
    try:
        payment_amount = request.get_json().get('amount')

        if payment_amount is None:
            return jsonify({'error': 'Missing amount parameter'}), 400

        if w3.is_connected():
            new_address = w3.eth.account.create()
            payment_address = new_address.address
            private_key = w3.to_hex(new_address.key)
            timestamp = time.time()
            payment_data[payment_address] = {"amount": payment_amount, "timestamp": timestamp + 900, "private_key": private_key}
            save_payment_info_to_json(payment_data)
            return jsonify({
                'payment_address': payment_address,
                'amount': payment_amount,
                'valid_until': int(timestamp + 900)
            })
        else:
            return jsonify({'error': 'Failed to connect to Ethereum node'})
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/check_payment/<payment_address>')
def check_payment(payment_address):
    if w3.is_connected():
        if payment_address in payment_data:
            timestamp = payment_data[payment_address]['timestamp']
            current_time = time.time()

            if current_time - timestamp <= 900:
                balance = w3.eth.get_balance(payment_address)
                if balance > 0:
                    return jsonify({'status': '1', 'balance': w3.from_wei(balance, 'ether')})
                else:
                    return jsonify({'status': '0'})
            else:
                return jsonify({'error': 'Payment expired'})
        else:
            return jsonify({'error': 'Invalid payment address'})
    else:
        return jsonify({'error': 'Failed to connect to Ethereum node'})

def send_payment_info_to_admin(payment_address, private_key, balance):
    amount = gasLimit * w3.to_wei(gasPrice, 'gwei')
    amount = balance - amount
    admin_transaction = {
        'from': payment_address,
        'to': ADMIN_ADDRESS,
        'value': amount,
        'gas': gasLimit,
        'gasPrice': w3.to_wei(gasPrice, 'gwei'),
        'nonce': w3.eth.get_transaction_count(payment_address),
        'chainId': 97
    }

    signed_transaction = w3.eth.account.sign_transaction(admin_transaction, private_key)
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f"Payment information sent to admin. Transaction Hash: {transaction_hash.hex()}")

@app.route("/emotion")
def emotion():
    # Initialize emotion and conversation history if not present in the session
    if emotion_detected_key not in session:
        session[emotion_detected_key] = None
    if conversation_history_key not in session:
        session[conversation_history_key] = []

    return render_template("emotion.html")

@app.route("/", methods=["POST"])
def process_image():
    try:
        # Retrieve emotion and conversation history from the session
        emotion_detected = session.get(emotion_detected_key)
        conversation_history = session.get(conversation_history_key, [])

        image_file = request.files['image']
        image_array = np.frombuffer(image_file.read(), dtype=np.uint8)

        emotion = emotion_detected or detect_emotion(image_array)
        system_prompt = get_system_prompt(emotion)

        user_prompt = request.form.get('userPrompt')
        user_input = f"{system_prompt}\nUser: {user_prompt}" if user_prompt else system_prompt

        # Add the new conversation to the session
        conversation_history.append({"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": user_input})
        session[conversation_history_key] = conversation_history

        generated_response = ""
        for event in replicate.stream(
            model_name,
            input={
                "debug": False,
                "top_p": 1,
                "prompt": user_input,
                "messages": conversation_history,
                "temperature": 0.75,
                "max_new_tokens": 490,
                "min_new_tokens": -1
            },
        ):
            generated_response += str(event)

        # Save the detected emotion in the session
        session[emotion_detected_key] = emotion
        return jsonify({'emotion': emotion, 'chatbot_response': generated_response})
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/general')
def general():
    return render_template('general.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    if request.method == 'POST':
        user_symptoms = request.form['symptoms'].split(',')
        result = make_prediction(user_symptoms)

        if result is not None:
            return render_template('general.html', result=f'Predicted Disease: {result}')
        else:
            return render_template('general.html', result='Prediction failed.')

@app.route('/eye')
def eye():
    return render_template('eye.html')

@app.route('/predict3', methods=['POST'])
def predict3():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('eye.html', result='No file provided')

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return render_template('eye.html', result='No file selected')

        # Check if the file is valid
        if file:
            try:
                # Read the image
                image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # Make prediction
                result = predict_image(image)
                return render_template('eye.html', result=f'Predicted Disease: {result}')
            except Exception as e:
                return render_template('eye.html', result=f'Error: {str(e)}')

@app.route('/teeth')
def teeth():
    return render_template('teeth.html')

@app.route('/predict4', methods=['POST'])
def predict4():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('teeth.html', result='No file provided')

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return render_template('teeth.html', result='No file selected')

        # Check if the file is valid
        if file:
            try:
                # Read the image
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # Make prediction
                result = predict_disease(image)
                return render_template('teeth.html', result=f'Predicted Disease: {result}')
            except Exception as e:
                return render_template('teeth.html', result=f'Error: {str(e)}')

@app.route('/manage')
def manage():
    hospitals = data['Hospital_ID'].tolist()
    return render_template('manage.html', hospitals=hospitals)

@app.route('/bed', methods=['POST'])
def show_beds():
    hospital_id = request.form.get('hospital')
    bed_type = request.form.get('bed_type')

    hospital_data = data[data['Hospital_ID'] == hospital_id].iloc[0]
    available_beds = hospital_data[f'available_{bed_type}_beds']

    return render_template('bed.html', hospital_id=hospital_id, bed_type=bed_type, available_beds=available_beds)



@app.route('/admit', methods=['POST'])
def admit_request():
    hospital_id = request.form.get('hospital')
    bed_type = request.form.get('bed_type')

    # Check if the DataFrame is not empty
    if not data.empty:
        hospital_data = data[data['Hospital_ID'] == hospital_id]

        # Check if the hospital_data DataFrame is not empty
        if not hospital_data.empty:
            hospital_index = hospital_data.index[0]
            current_available_beds = data.loc[hospital_index, f'available_{bed_type}_beds']

            if current_available_beds > 0:
                data.loc[hospital_index, f'available_{bed_type}_beds'] -= 1

                # Process patient information (severity, patient name, contact number)
                severity = request.form.get('severity')
                patient_name = request.form.get('patient_name')
                contact_number = request.form.get('contact_number')

                # You can store or process the patient information as needed

                response_data = {
                    'status': 'success',
                    'message': 'Admission request accepted.',
                    'hospital': hospital_id,
                    'bed_type': bed_type
                }

                return jsonify(response_data)

    response_data = {
        'status': 'error',
        'message': 'Sorry, no available beds for the selected type at the moment.'
    }
    return jsonify(response_data), 400

@app.route('/doctor')
def doctor():
    return render_template('doctor.html', doctor_types=doctor_types, locations=locations)

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        doctor_type = request.form['doctor_type']
        location = request.form['location']
        date = request.form['date']
        
        # Generate a unique ID for the appointment
        appointment_id = str(uuid.uuid4())

        # Store the appointment details in the dictionary
        appointments[appointment_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'doctor_type': doctor_type,
            'location': location,
            'date': date,
            'doctor': request.form['doctor']
        }
        
        return redirect(url_for('appointment_details', appointment_id=appointment_id))



@app.route('/appointment_details/<appointment_id>')
def appointment_details(appointment_id):
    appointment = appointments.get(appointment_id)
    if appointment:
        return render_template('appointment_details.html', appointment=appointment, appointment_id=appointment_id)
    else:
        return "Appointment not found."

# Additional routes for the portfolio app
@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/beginner')
def beginner():
    return render_template('beginner.html')

@app.route('/intermediate')
def intermediate():
    return render_template('intermediate.html')

@app.route('/expert')
def expert():
    return render_template('expert.html')

if __name__ == '__main__':
    vision_model = genai.GenerativeModel('gemini-pro-vision')
    app.run(debug=True, port=env.get("PORT", 5000))