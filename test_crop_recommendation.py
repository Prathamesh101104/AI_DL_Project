import numpy as np
import pickle

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    rfc = pickle.load(f)

with open('minmaxscaler.pkl', 'rb') as f:
    ms = pickle.load(f)

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    prediction = rfc.predict(transformed_features)
    return prediction[0]

# Crop mapping 
crop_num = {
    'Rice': 1,
    'Maize': 2,
    'Jute': 3,
    'Cotton': 4,
    'Coconut': 5,
    'Papaya': 6,
    'Orange': 7,
    'Apple': 8,
    'Muskmelon': 9,
    'Watermelon': 10,
    'Grapes': 11,
    'Mango': 12,
    'Banana': 13,
    'Pomegranate': 14,
    'Lentil': 15,
    'Blackgram': 16,
    'MungBean': 17,
    'MothBeans': 18,
    'PigeonPeas': 19,
    'KidneyBeans': 20,
    'ChickPea': 21,
    'Coffee': 22
}

# Create reverse mapping
crop_num_to_name = {v: k for k, v in crop_num.items()}

# Test with the same inputs from the notebook
N = 60
P = 55
k = 44
temperature = 23.0
humidity = 82
ph = 7.8
rainfall = 263

print("Testing Crop Recommendation System")
print("=" * 40)
print(f"Input Parameters:")
print(f"Nitrogen: {N}")
print(f"Phosphorus: {P}")
print(f"Potassium: {k}")
print(f"Temperature: {temperature}")
print(f"Humidity: {humidity}")
print(f"pH: {ph}")
print(f"Rainfall: {rainfall}")
print("-" * 40)

predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)
print(f"Predicted crop number: {predict}")

if predict in crop_num_to_name:
    crop_name = crop_num_to_name[predict]
    print(f"✅ {crop_name} is a best crop to be cultivated")
else:
    print("❌ Sorry, unable to recommend a proper crop for this environment")

print("=" * 40)

# Test with different inputs
print("\nTesting with different inputs:")
test_inputs = [
    (90, 42, 43, 20.88, 82.0, 6.5, 202.9),  
    (85, 58, 41, 21.77, 80.3, 7.04, 226.7),  
    (60, 55, 44, 23.0, 82.3, 7.84, 264.0),   
]

for i, (n, p, k, temp, hum, ph_val, rain) in enumerate(test_inputs, 1):
    pred = recommendation(n, p, k, temp, hum, ph_val, rain)
    crop = crop_num_to_name.get(pred, "Unknown")
    print(f"Test {i}: {crop} (Number: {pred})")
