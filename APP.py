import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Create synthetic PIT feature dataset
np.random.seed(42)
n_samples = 500

data = {
    'max_velocity': np.random.normal(4, 1, n_samples),
    'signal_duration': np.random.normal(20, 5, n_samples),
    'reflected_peaks': np.random.randint(0, 5, n_samples),
    'energy': np.random.normal(80, 20, n_samples),
    'defect_depth': np.random.uniform(0, 10, n_samples)
}

# Simulated labels: 0 = Good, 1 = Possible Defect, 2 = Defective
labels = []

for i in range(n_samples):
    if data['reflected_peaks'][i] >= 3 or data['defect_depth'][i] > 6:
        labels.append(2)  # Defective
    elif data['reflected_peaks'][i] == 2:
        labels.append(1)  # Possible defect
    else:
        labels.append(0)  # Good

df = pd.DataFrame(data)
df['condition'] = labels

# Step 2: Train-test split
X = df.drop(columns='condition')
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Good", "Possible Defect", "Defective"]))

# Step 5: Visualize predictions
plt.figure(figsize=(6, 4))
plt.hist(y_pred, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', rwidth=0.8)
plt.xticks([0, 1, 2], ["Good", "Possible Defect", "Defective"])
plt.title("Predicted Pile Conditions")
plt.xlabel("Condition")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
