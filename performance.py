import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
test_dir = 'dataset/test'
model_path = 'models/emotion_model1.keras'

# Image size and batch
img_size = 48
batch_size = 128

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# Load trained model
model = load_model(model_path)

# Predict
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# === PERFORMANCE METRICS ===
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


# === EVALUATION METRIC ===
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision (macro-average)
precision = precision_score(y_true, y_pred, average='macro')
print(f"Precision (macro): {precision:.4f}")

# Recall (macro-average)
recall = recall_score(y_true, y_pred, average='macro')
print(f"Recall (macro): {recall:.4f}")

# F1 Score (macro-average)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score (macro): {f1:.4f}")
