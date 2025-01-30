## This file is for reading the fixed set of data we collected with Bulut 
# and having the training model understand the different columns 
# proximal and distal difference and know based on those parameters the direction the glove is facing

import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Load your training data
data = pd.read_csv("training_dataset.csv")  # Ensure file exists and has proper data

# Step 2: Preprocess the data
X = data.drop(columns=["Direction"])
y = data["Direction"]

# Encode categorical features
categorical_columns = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])

# Encode target variable
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Save the model and encoders
joblib.dump(pipeline, 'glove_direction_model.pkl')
joblib.dump(y_encoder, 'label_encoder.pkl')
print("Model and encoders saved.")

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()