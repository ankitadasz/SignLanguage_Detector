import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Example dummy data (replace with your real data)
# Each sample = 42 values (21 landmarks × x,y)
X = []
y = []

# Example: 100 samples
for i in range(100):
    X.append(np.random.rand(42))   # fake landmark data
    y.append("HELLO")              # label

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")