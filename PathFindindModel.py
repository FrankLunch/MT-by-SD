from keras.models import load_model
import numpy as np
from PIL import Image
import sys
import os
import cv2
import time

import keras.backend as K

# Define the RMSE metric
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Load the model with custom_objects
from keras.models import load_model


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_model_from_path(model_path, custom_objects=None):
    """
    Load a Keras model from the given path.
    
    Args:
        model_path (str): Path to the model file.
        custom_objects (dict): Custom objects required for loading the model (optional).
    
    Returns:
        model: Loaded Keras model.
    """
    if custom_objects:
        model = load_model(model_path, custom_objects=custom_objects)
    else:
        model = load_model(model_path)
    return model

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    image = crop(image)  # 移除不必要部分
    image = resize(image)  # 调整到目标大小
    image = rgb2yuv(image)  # 转换颜色空间
    return image

import tensorflow as tf

def preprocess(image):
    image = tf.image.crop_to_bounding_box(image, 60, 0, 75, 320)  # 裁剪
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])   # 调整大小
    image = tf.image.rgb_to_yuv(image)                           # 转换颜色空间
    return image.eval(session=tf.compat.v1.Session())  # 使用 TensorFlow 1.x 的 Session


def predict_steering_angle(model, image):
    start_time = time.time()
    # Convert PIL image to NumPy array and preprocess it
    image_array = np.asarray(image)
    preprocessed_image = preprocess(image_array)  # Apply preprocessing
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    end_image_processing = time.time()
    # Predict the steering angle
    steering_angle = float(model.predict(preprocessed_image, batch_size=1))
    end_predict_time = time.time()
    print(f'image processing time is {end_image_processing-start_time} seconds, and predict time is {end_predict_time-end_image_processing} seconds')
    return steering_angle

import time
start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # use specific GPU
#requirement
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3

# load model
#model_path = "/data/tony/dave-2/model/epoch-dataset5-304.h5"
model_path = "/data/tony/dave-2/model/chauffeur-315.h5"
with HiddenPrints():
	#model = load_model_from_path(model_path)
    model = load_model(model_path, custom_objects={'rmse': rmse})

end_load = time.time()
load_time = end_load-start
print(f'load it by {load_time} seconds')
# 

image_path = "/data/tony/dave-2/IMG/2019_08_18_16_16_39_604.jpg"
image = Image.open(image_path)
end_imageopen = time.time()
image_open = end_imageopen-end_load
print(f'image opened by {image_open} seconds')
print('image , good')
# 
steering_angle = predict_steering_angle(model, image)
end_predict = time.time()
predict_time = end_predict - end_imageopen
print(f'GOOD , the prediction resul came by {predict_time} seconds')
print("Predicted Steering Angle:", steering_angle)
end = time.time()
run_time = end-start
print(f"total run time is {run_time}")


import cv2
import numpy as np

# Load the image
image = cv2.imread(image_path)

# Define the starting point (e.g., bottom center of the image)
height, width = image.shape[:2]
x_start = width // 2
y_start = height - 50  # Slightly above the bottom edge

# Define the arrow length
L = 30  # Adjust as needed

# Predicted steering angle in radians
theta = steering_angle

# Calculate the endpoint
x_end = int(x_start + L * np.sin(theta))
y_end = int(y_start - L * np.cos(theta))

# Draw the arrow
body_thickness = 2
head_scale = 1.5
cv2.arrowedLine(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 5, body_thickness, tipLength=head_scale/L)

# Save the image
cv2.imwrite('steered_image.png', image)
print('steered_image saved')

