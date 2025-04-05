import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import pandas as pd

dataset = pd.read_csv(r"C:\Users\dushy\Downloads\archive\Final Dataset\dataset.csv")

dataset['label'] = dataset['label'].map({'Real': 0, 'Fake': 1})

dataset['path'] = 'C:\\Users\\dushy\\Downloads\\archive\\Final Dataset\\' + dataset['path'].str.replace('/kaggle/input/stylegan-and-stylegan2-combined-', '')

print(dataset.head())
import os

dataset['path'] = dataset['path'].apply(lambda x: x.replace("dataset/Final Dataset/", ""))

invalid_paths = [path for path in dataset['path'] if not os.path.exists(path)]

if invalid_paths:
    print(f"Found {len(invalid_paths)} invalid image paths!")
    print(invalid_paths[:5]) 
else:
    print("All image paths are valid!")
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(img_path):
    try:
        img = cv2.imread(img_path) 
        img = cv2.resize(img, (299, 299)) 
        img = img / 255.0 
        return img
    except:
        return None  
images = []
labels = []
for index, row in dataset.iterrows():
    img = load_and_preprocess_image(row['path'])
    if img is not None:
        images.append(img)
        labels.append(row['label'])

X = np.array(images, dtype=np.float32)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print("Done")
