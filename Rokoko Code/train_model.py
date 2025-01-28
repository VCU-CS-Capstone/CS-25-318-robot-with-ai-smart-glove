import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your labeled data (replace 'your_training_data.csv' with your actual file)
data = pd.read_csv('your_training_data.csv')

# Features and target variable
X = data[['rotation_x', 'rotation_y', 'rotation_z', 'rotation_w']]  # Adjust columns as needed
y = data['direction']  # This should be the label column with directions

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model
model = LogisticRegression()  # You can also try other classifiers like RandomForest, SVM, etc.
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, 'finger_direction_model.joblib')
