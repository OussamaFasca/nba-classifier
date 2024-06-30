from flask import Flask, request, jsonify
import joblib

# Create a Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('best_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Make prediction using model loaded from disk
    prediction = model.predict([data['features']])
    
    # Take the first value of prediction
    output = prediction[0]
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True,host='0.0.0.0')