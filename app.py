from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and preprocessing objects
theta = joblib.load("model_theta.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("polynomial_features.pkl")

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data
    data = request.json
    input_data = np.array(data["features"]).reshape(1, -1)

    # Preprocess the input
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)

    # Make prediction
    prediction = (sigmoid(input_poly @ theta) >= 0.5).astype(int)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
