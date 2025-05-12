import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("heart.csv")
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# 2. Label Encoding for binary columns (e.g., 'Sex', 'ExerciseAngina')
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])  # M -> 1, F -> 0
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])  # Y -> 1, N -> 0

# 3. One-Hot Encoding for multi-class columns (e.g., 'ChestPainType', 'ST_Slope')
df = pd.get_dummies(df, columns=['ChestPainType', 'ST_Slope'], drop_first=True)

# 4. Label Encoding for non-numeric columns like 'RestingECG' or other categorical columns
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])  # 'Normal' -> 0, 'ST' -> 1, 'LVH' -> 2

# 5. Split features and label
print("\nSplitting dataset into features and target...")
X = df.drop("HeartDisease", axis=1)  # Features (all columns except 'HeartDisease')
y = df["HeartDisease"]  # Target (the 'HeartDisease' column)

# 6. Split into train/test
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 7. Scale features
print("\nScaling the features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data, then transform it
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler
print("Features scaled successfully!")

# 8. Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete!")

# 9. Test accuracy
print("\nEvaluating model on test data...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- Predict for new input ---
# Example input (make sure this matches the number of columns after preprocessing)
input_data = {
    "Age": 63,
    "Sex": 1,  # M -> 1, F -> 0
    "RestingBP": 145,
    "Cholesterol": 233,
    "FastingBS": 1,  # 1 -> True, 0 -> False
    "RestingECG": 0,  # 'Normal' -> 0, 'ST' -> 1, 'LVH' -> 2
    "MaxHR": 150,
    "ExerciseAngina": 0,  # N -> 0, Y -> 1
    "Oldpeak": 2.3,
    "ChestPainType_ATA": 1,  # One-hot encoding for ChestPainType
    "ChestPainType_NAP": 0,
    "ChestPainType_ASY": 0,  # Include all possible one-hot encoded columns
    "ST_Slope_Flat": 0,  # One-hot encoding for ST_Slope
    "ST_Slope_Up": 1
}

# Convert the dictionary to a DataFrame to match the input format
input_df = pd.DataFrame([input_data])

# Ensure the input data has the same columns as the training data
input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

# 10. Scale the new input data using the same scaler
input_scaled = scaler.transform(input_df)
print(f"Scaled input data: {input_scaled}\n")

# 11. Make the prediction
prediction = model.predict(input_scaled)

# 12. Show the result of the prediction
if prediction[0] == 1:
    print("✅ Prediction: Person is likely to have heart disease.")
else:
    print("❌ Prediction: Person is unlikely to have heart disease.")
