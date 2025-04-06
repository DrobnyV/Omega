import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('../training_data_2023_2024.csv')

print("Unique Result values:")
print(df['Result'].unique())

result_mapping = {
    "Draw": 0,
    "Home Win": 1,
    "Away Win": 2
}
df['Result_Label'] = df['Result'].map(result_mapping)

print("Unique Result_Label values:")
print(df['Result_Label'].unique())

if df['Result_Label'].isna().any():
    raise ValueError("Result_Label contains NaN values. Check Result column mapping.")

feature_cols = [
    'Home_Wins', 'Home_Draws', 'Home_Losses', 'Home_PtsPerMatch', 'Home_GoalDiff', 'Home_xGD',
    'Away_Wins', 'Away_Draws', 'Away_Losses', 'Away_PtsPerMatch', 'Away_GoalDiff', 'Away_xGD'
]
X = df[feature_cols].fillna(0)
y = df['Result_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Logistic Regression ---
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# --- Model 2: Random Forest Classifier ---
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# --- Model 3: Custom Neural Network ---
print("\nTraining Neural Network...")
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
nn_loss, nn_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
nn_pred = np.argmax(nn_model.predict(X_test_scaled, verbose=0), axis=1)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, nn_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# --- Model 4: XGBoost ---
from xgboost import XGBClassifier
print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# --- Model 5: SVM ---
from sklearn.svm import SVC
print("\nTraining Support Vector Machine...")
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, svm_pred, target_names=['Draw', 'Home Win', 'Away Win']))




print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

scaler_filename = '../scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")
svm_filename = '../svm_model.pkl'
joblib.dump(svm_model, svm_filename)
print(f"SVM model saved to {svm_filename}")