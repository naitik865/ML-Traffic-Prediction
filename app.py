from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# Load the trained model
with open('vehicle_count_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mappings for categorical variables
day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4}
weather_map = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2}

# Home route for root URL
@app.route('/')
def home():
    return '''
    <h2>Vehicle Count Prediction API</h2>
    <p>Use the <code>/predict</code> endpoint with query parameters.</p>
    <p>Example: <code>/predict?hour=9&day=Tue&weather=Rainy</code></p>
    '''

# Optional: Handle favicon.ico requests to avoid 404s
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )

# Prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    try:
        hour = int(request.args.get('hour', 8))
        day = request.args.get('day', 'Mon')
        weather = request.args.get('weather', 'Clear')

        input_data = [[hour, day_map.get(day, 0), weather_map.get(weather, 0)]]
        pred = model.predict(input_data)[0]
        return jsonify({'predicted_vehicle_count': int(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
