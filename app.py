from flask import Flask, render_template, request,flash,redirect, url_for
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

import os


app = Flask(__name__)
img = os.path.join('static')
app.secret_key = '98345876252'
@app.route('/')
def home():
    return render_template('index.html')

# Route for the image upload
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the post request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    img_bytes = file.read()

    img_np = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img_sift = cv2.drawKeypoints(img, keypoints, None)

    edges = cv2.Canny(gray, 100, 200)

    img_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    result = np.concatenate((img_sift, img_edge), axis=1)

    cv2.imwrite('static/result.png', result)

    return render_template('result.html', result=file)

if __name__ == '__main__':
    app.run(debug=False)