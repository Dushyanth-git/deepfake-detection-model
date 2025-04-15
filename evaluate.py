import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Validation split (same as training)
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = load_model('base_model.h5')

# Predict
y_pred_probs = model.predict(X_val, batch_size=32)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Metrics
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"{'Metric':<12} | {'Value'}")
print(f"{'-'*25}")
print(f"{'Accuracy':<12} | {acc:.4f}")
print(f"{'Precision':<12} | {prec:.4f}")
print(f"{'Recall':<12} | {rec:.4f}")
print(f"{'F1-Score':<12} | {f1:.4f}\n")

print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
