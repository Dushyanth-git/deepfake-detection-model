import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

model = load_model('base_model.h5')
 
def pre_process(img_path):
    img = image.load_img(img_path, target_size=(299,299))
    image_array = image.img_to_array(img)
    img_array = np.expand_dims(image_array, axis=0)
    img_array  = img_array/255.0
    return img_array
img_path = r"C:\Users\dushy\Desktop\avatar-genffba84391dcda8dacd645c299eaa55bb.jpg"
img = pre_process(img_path)
predictions = model.predict(img)
print(predictions)
if predictions[0][0] > 0.5:
    print("Deepfake detected")
else:
    print("Real thing")



