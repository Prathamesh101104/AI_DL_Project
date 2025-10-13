from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pandas as pd
import sklearn
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# importing model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    ms = None

# creating flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages


# Crop image mapping for real crop images (matching actual filenames)
crop_images = {
    "Rice": "rice.jpg",
    "Maize": "maize.jpg", 
    "Jute": "Jute.jpg",  # Fixed case
    "Cotton": "cotton.jpg",
    "Coconut": "coconut.jpg",
    "Papaya": "Papaya.jpg",  # Fixed case
    "Orange": "orange.jpg",
    "Apple": "apple.jpg",
    "Muskmelon": "Muskmelon.jpg",  # Fixed case
    "Watermelon": "watermelon.jpg",
    "Grapes": "grapes.jpg",
    "Mango": "mango.jpg",
    "Banana": "banana.jpg",
    "Pomegranate": "pomegrante.jpg",  # Fixed filename (actual file has typo)
    "Lentil": "lentil.jpg",
    "Blackgram": "blackgram.jpg",
    "Mungbean": "mungbean.jpg",
    "Mothbeans": "mothbeans.jpg",
    "Pigeonpeas": "pigeonbeans.jpg",  # Fixed filename (actual file is pigeonbeans)
    "Kidneybeans": "kidneybeans.jpg",
    "Chickpea": "chickpea.jpg",
    "Coffee": "Coffee.jpeg",  # Fixed case and extension
    "Unknown": "rice.jpg"  # Using rice.jpg as fallback since no default_crop.jpg exists
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Input validation
        if not all([N >= 0, P >= 0, K >= 0, -10 <= temp <= 50, 0 <= humidity <= 100, 0 <= ph <= 14, rainfall >= 0]):
            return render_template('index.html', error="Please enter valid values for all parameters.")

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Check if models are loaded
        if model is None or ms is None:
            return render_template('index.html', error="Model not available. Please try again later.")

        scaled_features = ms.transform(single_pred)
        prediction = model.predict(scaled_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there"
            crop_image = crop_images.get(crop, "default_crop.jpg")
            
            # Log the prediction
            logger.info(f"Prediction made: {crop} for parameters: N={N}, P={P}, K={K}, Temp={temp}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")
            
        else:
            crop = "Unknown"
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = crop_images.get(crop, "default_crop.jpg")
            logger.warning(f"Unknown prediction value: {prediction[0]}")

        return render_template('index.html', result=result, crop=crop, crop_image=crop_image)

    except ValueError as e:
        logger.error(f"ValueError in prediction: {e}")
        return render_template('index.html', error="Please enter valid numeric values for all parameters.")
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return render_template('index.html', error="An unexpected error occurred. Please try again.")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

       