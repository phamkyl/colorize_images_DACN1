from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab, gray2rgb
from skimage.transform import resize
from PIL import Image
import io
import os

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('colorize_image012.h5', compile=False)

app = Flask(__name__)

# Thư mục để lưu trữ các ảnh đã được tô màu
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Hàm tiền xử lý ảnh
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Thay đổi kích thước ảnh
    image = np.array(image)
    if image.shape[2] == 4:  # Loại bỏ kênh alpha nếu có
        image = image[:, :, :3]
    lab = rgb2lab(image)  # Chuyển đổi ảnh từ RGB sang LAB
    l = lab[:, :, 0]  # Lấy kênh L
    l = l.reshape(1, 256, 256, 1)  # Định hình lại để phù hợp với đầu vào mô hình
    return l

# Hàm hậu xử lý ảnh
def postprocess_image(l, ab):
    lab = np.zeros((256, 256, 3))
    lab[:, :, 0] = l[0][:, :, 0]  # Đặt kênh L
    lab[:, :, 1:] = ab[0] * 128  # Đặt các kênh AB
    rgb = lab2rgb(lab)  # Chuyển đổi từ LAB sang RGB
    return (rgb * 255).astype(np.uint8)  # Chuyển đổi về định dạng uint8

@app.route('/')
def index():
    return render_template('index.html')  # Trả về giao diện chính

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:  # Kiểm tra nếu không có file trong request
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':  # Kiểm tra nếu không có file được chọn
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file).convert('RGB')  # Mở file ảnh và chuyển sang RGB
        l = preprocess_image(image)  # Tiền xử lý ảnh
        ab = model.predict(l)  # Dự đoán các kênh AB
        colorized_image = postprocess_image(l, ab)  # Hậu xử lý để tạo ảnh đã tô màu
        colorized_image_pil = Image.fromarray(colorized_image)  # Chuyển đổi sang định dạng PIL

        # Lưu ảnh đã tô màu
        output_path = os.path.join(output_dir, file.filename)
        colorized_image_pil.save(output_path)

        img_io = io.BytesIO()
        colorized_image_pil.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')  # Trả về ảnh đã tô màu
    except Exception as e:  # Xử lý lỗi
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng Flask ở chế độ debug
