from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import keras
import os
import cv2
import os
import requests
from tqdm import tqdm

import torch
from torchvision import models



from tensorflow.python.keras.utils.data_utils import get_file

# def loadModel(modelURL):
#
#     fileName = os.path.basename(modelURL)
#     modelName = fileName[:fileName.index('.')]
#
#     cacheDir = "./"
#
#     os.makedirs(cacheDir, exist_ok=True)
#
#     get_file(fname=fileName,origin=modelURL, cache_dir=cacheDir, extract=True)
#     return cacheDir+modelName+'.pth'


import os
import requests
from tqdm import tqdm


def download_model_checkpoint():
    # Set the model name and cache directory
    modelName = 'best_model_weights.h5'
    cacheDir = "./"

    # Ensure the cache directory exists, if not, create it
    os.makedirs(cacheDir, exist_ok=True)

    # Specify the path to check if the model file already exists
    checkpoint_path = os.path.join(cacheDir, modelName)

    # Check if the model file already exists
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Define the URL from which to download the model checkpoint
    url = "https://huggingface.co/Nechba/car-segmentation-intern/resolve/main/best_model_weights.h5"

    # Send a GET request to the URL with streaming enabled
    response = requests.get(url, stream=True)

    # Get the total size of the file
    total_size = int(response.headers.get('content-length', 0))

    # Set the block size for downloading (1 Kibibyte)
    block_size = 1024

    # Initialize a progress bar for tracking download progress
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Open a file for writing in binary mode
    with open(checkpoint_path, 'wb') as file:
        # Iterate over the content of the response in blocks
        for data in response.iter_content(block_size):
            # Update the progress bar and write the data to the file
            progress_bar.update(len(data))
            file.write(data)

    # Close the progress bar
    progress_bar.close()

    # Return the path to the downloaded checkpoint
    return checkpoint_path


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.io.decode_jpeg(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[512, 512])
    else:
        image = tf.io.decode_jpeg(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[512, 512])
    return image


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = (predictions > 0.5).astype(np.uint8)
    return predictions


def remove_background_internal(image_path):
    modelURL = "https://huggingface.co/Nechba/car-segmentation-intern/resolve/main/best_model_weights.h5"
    save_path = download_model_checkpoint()
    model = load_model(save_path)
    image_tensor = read_image(image_path)
    prediction_mask = infer(image_tensor=image_tensor, model=model)

    # Convert EagerTensor to NumPy array
    image_tensor_np = np.array(image_tensor)

    # Create a copy of the original image to modify
    result_image = np.copy(image_tensor_np)

    # Copy the car from the original image to the result image using the refined mask
    result_image[:, :, :3][prediction_mask == 1] = 255

    return result_image


def plot_image(image, figsize=(5, 3)):
    if image.shape[-1] == 3:
        plt.imshow(keras.utils.array_to_img(image))
    else:
        plt.imshow(image)
    plt.show()



