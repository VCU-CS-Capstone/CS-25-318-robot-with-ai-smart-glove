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

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import joblib

# # Step 1: Load your training data
# data = pd.read_csv("training_dataset.csv")  # Make sure this file exists with proper data

# # Step 2: Preprocess the data
# # Let's assume 'Direction' is the target column (what you're predicting)
# # All other columns are features
# X = data.drop(columns=["Direction"])  # Dropping non-feature columns
# y = data["Direction"]  # The target variable

# # Step 3: Encode categorical features (if any) using LabelEncoder
# # Here we handle encoding for any categorical features in X if they exist
# categorical_columns = X.select_dtypes(include=["object"]).columns
# encoder = LabelEncoder()

# for column in categorical_columns:
#     X[column] = encoder.fit_transform(X[column])

# # Step 4: Encode the target variable (Direction) using LabelEncoder
# y_encoder = LabelEncoder()
# y_encoded = y_encoder.fit_transform(y)

# # Step 5: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Step 6: Create a pipeline with preprocessing and training steps
# # We use StandardScaler to scale numerical data if needed
# # We can add more steps for feature engineering if needed
# pipeline = Pipeline(steps=[
#     ('scaler', StandardScaler()),  # Standardize the numerical features
#     ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForest classifier
# ])

# # Step 7: Train the model
# pipeline.fit(X_train, y_train)

# # Step 8: Evaluate the model
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Step 9: Save the trained model, scaler, and label encoder
# joblib.dump(pipeline, 'glove_direction_model.pkl')
# joblib.dump(y_encoder, 'label_encoder.pkl')  # Save the label encoder
# joblib.dump(pipeline.named_steps['scaler'], 'scaler.pkl')  # Save the scaler used for training
# print("Model, label encoder, and scaler saved.")
