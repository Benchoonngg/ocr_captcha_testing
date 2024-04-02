from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM
import keras


# - - - - - - - - - - -


# Load the Original Images Trained OCR model
ocr_model = load_model("/Volumes/Hocson - External Device/Coding/Fanta Tech/test 16 (custom notebook) (original images w: inv + gray)/ocr_model_OI.keras")

# Load the Inverted Trained OCR Model
#ocr_model = load_model("/Volumes/Hocson - External Device/Coding/Fanta Tech/test 17 (custom notebook) (inverted images w: inv + gray)/ocr_model.keras")

# Load a single image file
image_path = "/Volumes/Hocson - External Device/Coding/Fanta Tech/LTO_Captcha/1st batch/grayscale/1DETK.jpg"

# Check if the image file exists
if os.path.isfile(image_path):
    images = [image_path]
    labels = [os.path.basename(image_path).split(".jpg")[0]]
    characters = set(char for label in labels for char in label)

    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Numbers of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Batch size for training and validation
    batch_size = 1  # Since there's only one image

    # Desired image dimensions
    img_width = 580
    img_height = 160

    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolutional blocks and each block will have
    # 3 pooling layers which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    downsample_factor = 4

    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in labels])

    # Check if the image dimensions match the desired dimensions
    with Image.open(image_path) as img:
        if img.width != img_width or img.height != img_height:
            print(f'Image dimensions do not match: Width = {img.width}, Height = {img.height}')

    # Extract labels from the single image (if applicable)
    labels = [os.path.basename(image_path).split(".jpg")[0]]

    print(f'{labels=}')

    # Check for labels with different lengths
    different_sized_labels = [label for label in labels if len(label) != max_length]
    print(f'{different_sized_labels=}')

    assert len(different_sized_labels) == 0

else:
    print("Image file does not exist:", image_path)


# - - - - - - - - - - - -


# Mapping characters to integers
#char_to_num = layers.experimental.preprocessing.StringLookup(
    #vocabulary=list(characters), num_oov_indices=0, mask_token=None
#)
# Char to integers
char_to_num = layers.StringLookup(
	vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characteres
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_sample(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Decode image
    img = tf.io.decode_jpeg(img, channels=3)  # Decode with 3 channels for color images
    # Invert colors
    img = 1 - img  # Inversion by subtracting pixel values from 1
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Convert to grayscale
    img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale
    # Transpose the image because we want the time dimension to correspond to the width of the image
    img = tf.transpose(img, perm=[1, 0, 2])
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dictionary containing the image and its label
    return {"image": img, "label": label}




# - - - - - - - - - - - -


# Define LayerCTC class and register it for serialization
@tf.keras.utils.register_keras_serializable()
class LayerCTC(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Return computed predictions
        return y_pred


def model_build(img_width, img_height):
    # Define the inputs to the model
    input_img = Input(shape=(img_width, img_height, 1), name="image", dtype="float32")  # Added input layer for image
    img_labels = Input(name="label", shape=(None,), dtype="float32")  # Added input layer for labels

    # First convolutional block
    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    # Second convolutional block
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    # Reshaping the output before passing to RNN
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    x = Dropout(0.2)(x)

    # RNNs
    x = Bidirectional(LSTM(
        128, return_sequences=True, dropout=0.25))(x)
    x = Bidirectional(LSTM(
        64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = Dense(
        len(char_to_num.get_vocabulary()) + 1,
        activation="softmax", name="dense2"
    )(x)

    # Calculate CTC loss at each step
    output = LayerCTC(name="ctc_loss")(img_labels, x)

    # Defining the model with two inputs
    model = Model(
        inputs=[input_img, img_labels],
        outputs=output,
        name="ocr_model_v1"
    )
    opt = keras.optimizers.Adam()

    # Compile the model
    model.compile(optimizer=opt)

    return model

# Build the model with specified image dimensions
model = model_build(img_width=580, img_height=160)
model.summary()


# - - - - - - - - - - - -


# Get the prediction model by extracting layers till output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()

# A utility function to decode the output of the network
def decode_single_prediction(pred):
    input_len = np.array([pred.shape[1]])
    # Use greedy search. For complex tasks, you can use beam search
    result = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][0]
    # Get back the text from the result
    decoded_text = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    return decoded_text


# - - - - - - - - - - - -


train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

# - - - - - - - - - - - -

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Check the validation on a few samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}".replace('[UNK]', '_')
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()


# - - - - - - - - - - - -
