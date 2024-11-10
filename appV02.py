from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab, gray2rgb
from skimage.transform import resize
from PIL import Image
import io
import os

# Load the trained model
model = tf.keras.models.load_model('colorize_image012.h5',compile = False)

app = Flask(__name__)

# Directory to save the colorized images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)


def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[2] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    lab = rgb2lab(image)
    l = lab[:, :, 0]
    l = l.reshape(1, 256, 256, 1)
    return l


def postprocess_image(l, ab):
    lab = np.zeros((256, 256, 3))
    lab[:, :, 0] = l[0][:, :, 0]
    lab[:, :, 1:] = ab[0] * 128
    rgb = lab2rgb(lab)
    return (rgb * 255).astype(np.uint8)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file).convert('RGB')
        l = preprocess_image(image)
        ab = model.predict(l)
        colorized_image = postprocess_image(l, ab)
        colorized_image_pil = Image.fromarray(colorized_image)

        # Save the colorized image
        output_path = os.path.join(output_dir, file.filename)
        colorized_image_pil.save(output_path)

        img_io = io.BytesIO()
        colorized_image_pil.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
