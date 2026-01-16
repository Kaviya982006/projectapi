import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create or load dataset
# If you don't have instagram_data.csv, we'll generate a small sample dataset
try:
    df = pd.read_csv("instagram_data.csv")
except FileNotFoundError:
    data = {
        "followers": [50000, 20, 100, 3000, 50, 8000, 10, 15000, 60, 40000],
        "following": [200, 5000, 1000, 150, 2000, 300, 1000, 500, 2500, 100],
        "posts": [1200, 2, 0, 200, 5, 400, 1, 600, 3, 900],
        "is_private": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "label": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

# Step 2: Split features and labels
X = df[["followers", "following", "posts", "is_private"]]
y = df["label"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train XGBoost model
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Save model and scaler
# Save scaler with pickle
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save model with XGBoost native format (safer than pickle)
model.save_model("model.json")

print("Training complete. Files saved: scaler.pkl and model.json")
