import base64
import cv2
import io
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
from skimage.color import rgb2lab, lab2rgb

app = Flask(__name__)

# Load the pre-trained model
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('chuyennganh1_colorizeimagev0257.h5', compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError())

def preprocess_input_image(image):
    resized_image = cv2.resize(image, (256, 256))
    resized_image = resized_image.astype(float) / 255.0
    lab_image = rgb2lab(resized_image)
    l_channel = lab_image[:, :, 0] / 100.0
    return l_channel

def colorize_image(input_image):
    # Ensure input image is in RGB format
    if input_image.shape[2] == 4:  # Convert RGBA to RGB if necessary
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
    elif input_image.shape[2] == 1:  # Convert grayscale to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[2] != 3:  # Handle unexpected formats gracefully
        raise ValueError("Input image must be in RGB or RGBA format.")

    # Resize the input image to 256x256
    resized_image = cv2.resize(input_image, (256, 256))

    l_channel = preprocess_input_image(resized_image)

    # Colorize the image using the model
    ab_channels = model.predict(np.expand_dims(l_channel, axis=-1).reshape(1, *l_channel.shape, 1))[0]
    ab_channels = ab_channels * 128.0

    # Combine the L channel and colorized AB channels
    colorized_lab = np.zeros((256, 256, 3))
    colorized_lab[:, :, 0] = l_channel * 100.0
    colorized_lab[:, :, 1:] = ab_channels

    # Convert Lab image back to RGB color space
    colorized_rgb = lab2rgb(colorized_lab) * 255.0
    colorized_rgb = colorized_rgb.astype(np.uint8)

    return colorized_rgb

@app.route('/', methods=['GET', 'POST'])
def colorization():
    if request.method == 'POST':
        # Get the uploaded image from the request
        file = request.files['file']
        if file:
            try:
                # Read the image and convert it to numpy array
                img_array = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # Ensure image is loaded with correct channels

                # Colorize the image
                colorized_img = colorize_image(img)

                # Convert the colorized image to base64 string for displaying
                pil_image = Image.fromarray(colorized_img)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return render_template('index.html', colorized_img=img_base64)

            except Exception as e:
                error_message = f"Error colorizing the image: {str(e)}"
                app.logger.error(error_message)

                # Handle the error and display an error message
                return render_template('index.html', error_message=error_message)
        else:
            error_message = "No image uploaded."
            app.logger.error(error_message)
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
