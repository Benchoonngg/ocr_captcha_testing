import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

direc = Path("/Volumes/Hocson - External Device/Coding/Fanta Tech/LTO_Captcha/captcha_original")
#direc = Path("/Volumes/Hocson - External Device/Coding/Fanta Tech/Script/testing_one_image")
#direc = Path("/Volumes/Hocson - External Device/Coding/Fanta Tech/LTO_Captcha/1st batch/inverted")
#direc = Path("/Volumes/Hocson - External Device/Coding/Fanta Tech/LTO_Captcha/1st batch/bw")

extension = "jpg"
img_width = 580
img_height = 160
expected_max_length = 5
expected_min_length = 5

dir_img = sorted(list(map(str, list(direc.glob(f"*.{extension}")))))
img_labels = [img.split(os.path.sep)[-1].
			split(f".{extension}")[0] for img in dir_img]
char_img = set(char for label in img_labels for char in label)
char_img = sorted(list(char_img))

print(f"Number of dir_img found: {len(dir_img)}")
print(f"Number of img_labels found: {len(img_labels)}: {img_labels}")
print(f"Number of unique char_img: {len(char_img)}: {char_img}")

# Batch Size of Training and Validation
batch_size = 16


# Setting the Maximum Length
max_length = max([len(label) for label in img_labels])
if expected_max_length:
	unexpected_labels = [label for label in img_labels if len(label) > expected_max_length]
	assert [] == unexpected_labels, ','.join(unexpected_labels)
if expected_min_length:
	unexpected_labels = [label for label in img_labels if len(label) < expected_min_length]
	assert [] == unexpected_labels, ','.join(unexpected_labels)

# Char to integers
char_to_num = layers.StringLookup(
	vocabulary=list(char_img), mask_token=None
)

# Integers to original chaecters
num_to_char = layers.StringLookup(
	vocabulary=char_to_num.get_vocabulary(),
	mask_token=None, invert=True
)


def data_split(dir_img, img_labels,
			train_size=0.9, shuffle=True):
	# Get the total size of the dataset
	size = len(dir_img)
	# Create an indices array and shuffle it if required
	indices = np.arange(size)
	if shuffle:
		np.random.shuffle(indices)
	# Calculate the size of training samples
	train_samples = int(size * train_size)
	# Split data into training and validation sets
	x_train, y_train = dir_img[indices[:train_samples]], img_labels[indices[:train_samples]]
	x_valid, y_valid = dir_img[indices[train_samples:]], img_labels[indices[train_samples:]]
	return x_train, x_valid, y_train, y_valid


# Split data into training and validation sets
x_train, x_valid, y_train, y_valid = data_split(np.array(dir_img), np.array(img_labels))


def encode_sample(img_path, label):
	# Read the image
	img = tf.io.read_file(img_path)
	# Converting the image to grayscale
	img = tf.io.decode_png(img, channels=1)
	img = tf.image.convert_image_dtype(img, tf.float32)
	# Resizing to the desired size
	img = tf.image.resize(img, [img_height, img_width])
	# Transposing the image
	img = tf.transpose(img, perm=[1, 0, 2])
	# Mapping image label to numbers
	label = char_to_num(tf.strings.unicode_split(label,
												input_encoding="UTF-8"))

	return {"image": img, "label": label}

# ========

from PIL import Image
import os

for filename in os.listdir(direc):
    if filename.endswith('.jpeg'):
        # Construct the full path to the image
        filepath = os.path.join(direc, filename)

        # Open the image and print its width and height
        with Image.open(filepath) as img:
            if img.width != img_width or img.height != img_height:
                print(f'{filename}: Width = {img.width}, Height = {img.height}')

print(f'{img_labels=}')

different_sized_labels = [label for label in img_labels if len(label) != max_length]
print(f'{different_sized_labels=}')

assert len(different_sized_labels) == 0

# ========

# Creating training dataset
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_train = (
	dataset_train.map(
		encode_sample, num_parallel_calls=tf.data.AUTOTUNE
	)
	.batch(batch_size)
	.prefetch(buffer_size=tf.data.AUTOTUNE)
)


# Creating validation dataset
val_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
val_data = (
	val_data.map(
		encode_sample, num_parallel_calls=tf.data.AUTOTUNE
	)
	.batch(batch_size)
	.prefetch(buffer_size=tf.data.AUTOTUNE)
)


# ========

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM
import keras

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


# ========

# Build the model with specified image dimensions
model = model_build(img_width=580, img_height=160)
model.summary()

# ========

# Get the Model
prediction_model = keras.models.Model(
	model.get_layer(name="image").input,
	model.get_layer(name="dense2").output
)
prediction_model.summary()


def decode_batch_predictions(pred):
	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	results = keras.backend.ctc_decode(pred,
									input_length=input_len,
									greedy=True)[0][0][
		:, :max_length
	]
	output_text = []
	for res in results:
		res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
		output_text.append(res)
	return output_text

# ========

# From original prediction, using the ocr model instead of the validation batch

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the OCR model from the specified path
ocr_model = load_model("ocr_model_OI.keras", compile=False)

# From original prediction, using the ocr model instead of the validation batch

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

try:
    # Iterate through the validation batch
    for batch in val_data.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        # Make predictions using the loaded model
        preds = ocr_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        # Visualize the original images along with their corresponding predicted texts
        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}".replace('[UNK]', '_')
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
    plt.show()

except Exception as e:
    print("Error Encountered:", e)
    print("Failed to visualize predictions.")

    


except Exception as e:
    print("Error Encountered:", e)
    print("Failed to visualize predictions.")




