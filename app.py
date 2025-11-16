# app.py
import io
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

DATA = joblib.load("savedmodel.pth")
MODEL = DATA['model']

HTML = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload a face image (64x64 grayscale)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h2>Predicted class: {{ pred }}</h2>
{% endif %}
"""

def preprocess_image(file_stream):
    # convert uploaded image to grayscale 64x64 flattened vector normalized like Olivetti dataset
    img = Image.open(file_stream).convert('L')  # grayscale
    img = img.resize((64,64))
    arr = np.array(img, dtype=np.float32)
    # Olivetti images are scaled 0..1
    arr = arr / 255.0
    flat = arr.flatten()
    return flat.reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template_string(HTML, pred="No file uploaded")
        try:
            vec = preprocess_image(file)
            label = MODEL.predict(vec)[0]
            pred = int(label)
        except Exception as e:
            pred = f"Error: {e}"
    return render_template_string(HTML, pred=pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
