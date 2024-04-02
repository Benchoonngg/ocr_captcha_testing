from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the OCR model
ocr_model = load_model("ocr_model_OI.keras")

# Define the expected input dimensions
img_width = 580
img_height = 160  # Update the height to match the model's input shape

# Define the characters including a default character for unknown characters
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
default_char = '_'
characters_with_default = characters + default_char

# Define the mapping function including the default character
num_to_char = {i: char if i < len(characters) else default_char for i, char in enumerate(characters_with_default)}

# Create char_to_num mapping by reversing num_to_char
char_to_num = {char: num for num, char in num_to_char.items()}

def decode_batch_predictions_single(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]  # Using width as input length
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res_str = ''.join([num_to_char[int(r)] for r in res if int(r) != -1])
        output_text.append(res_str)
    return output_text

def preprocess_single_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((img_width, img_height))  # Resize the image
    image = np.array(image)  # Convert PIL image to numpy array
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.transpose(image, axes=(1, 0, 2))  # Transpose the image
    image = image / 255.0  # Normalize pixel values
    return image


# Function to predict text from a single image
def predict_single_image(image_path):
    preprocessed_image = preprocess_single_image(image_path)
    predictions = ocr_model.predict(np.expand_dims(preprocessed_image, axis=0))
    print(predictions)
    predicted_text = decode_batch_predictions_single(predictions)
    return predicted_text

# Path to the single image
image_path = '226y1.jpg'

# Predict text from the single image
predicted_text = predict_single_image(image_path)
print("Predicted Text:", predicted_text)
