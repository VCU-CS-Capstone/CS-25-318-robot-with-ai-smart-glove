import pandas as pd
import joblib

# Step 1: Load the trained model, scaler, and label encoder
model = joblib.load('glove_direction_model.pkl')  # Load the trained model
scaler = joblib.load('scaler.pkl')  # Load the scaler used during training
y_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder used during training

# Step 2: Load the new data (make sure it follows the same structure as your training data)
new_data = pd.read_csv('new_dataset_v1.csv')  # Replace with your actual new data file

# Step 3: Preprocess the new data (scale the features)
X_new = new_data.drop(columns=['Direction'], errors='ignore')  # Drop 'Direction' if it exists

# Ensure categorical features are encoded the same way as during training
categorical_columns = X_new.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    X_new[column] = y_encoder.transform(X_new[column])

# Apply the same scaling as during training
X_new_scaled = scaler.transform(X_new)  # Apply the same scaling as during training

# Step 4: Make predictions
predictions = model.predict(X_new_scaled)  # Get the predicted directions (numeric)

# Step 5: Map predictions back to human-readable labels
# Use the same label encoder to convert numeric predictions back to the original labels
predicted_directions = y_encoder.inverse_transform(predictions)

# Step 6: Add predictions to the original data
new_data['Predicted_Direction'] = predicted_directions  # Add the predictions to the new data
print(new_data[['Predicted_Direction']])  # Print out the predictions

# Optionally, you can save the predictions to a new file
new_data.to_csv('predicted_directions.csv', index=False)
print("Predictions saved to 'predicted_directions.csv'")
