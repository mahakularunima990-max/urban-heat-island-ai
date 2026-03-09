import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Select features
X = data[['temperature','ndvi','building_density','population']]

# Target variable
y = data['heat_risk']

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully and saved as model.pkl")