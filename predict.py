import pickle

# Load model
with open('vehicle_count_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_vehicle_count(hour, day_of_week, weather):
    day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4}
    weather_map = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2}
    input_data = [[hour, day_map[day_of_week], weather_map[weather]]]
    return model.predict(input_data)[0]

if __name__ == "__main__":
    count = predict_vehicle_count(8, 'Mon', 'Clear')
    print(f"Predicted vehicle count: {int(count)}")