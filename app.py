from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pickle  # For loading the scaler

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("linear_regression_model.pkl")  # Replace with the correct path

# Load the scaler
with open("scaler.pkl", "rb") as file:  # Replace with the correct path
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = int(request.form['Age'])
        experience = int(request.form['Experience'])
        salary = int(request.form['Salary'])

        # Calculate the Age-to-Experience ratio
        age_experience_ratio = age / (experience + 1)

        # Prepare the input data
        input_data = np.array([[age, experience, salary, age_experience_ratio]])

        # Standardize the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the loaded model
        predicted_price = model.predict(input_data_scaled)[0]

        return jsonify({"Predicted House Price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)