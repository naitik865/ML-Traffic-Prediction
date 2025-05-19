import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('simulated_vehicle_data.csv')

# Encode categorical columns
df['day_of_week'] = LabelEncoder().fit_transform(df['day_of_week'])
df['weather'] = LabelEncoder().fit_transform(df['weather'])

X = df[['hour', 'day_of_week', 'weather']]
y = df['vehicle_count']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('vehicle_count_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved.")