import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model('Updated_pneumonia_detection_model.h5')  # Load the saved model


def predict_image(img_path):
  img = image.load_img(img_path, target_size=(150, 150))   # Resize image
  img_array = image.img_to_array(img) / 255.0   # Convert to array and normalize
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

  prediction = model.predict(img_array)

  if prediction[0] > 0.5:
        print('Pneumonia')
  else:
        print('Normal')

if __name__ == '__main__':
    img_path = 'your.jpg'  # Image path
    predict_image(img_path)
