import numpy as np
import os
import keras
import base64
import io
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from flask import request
from flask import Flask
from flask import jsonify
from aeye_labels import labels
# from captionbot import CaptionBot


app = Flask(__name__)


def get_model():
    global model
    model = load_model('aeye.h5')
    print("Model Loaded!")


def prepare_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array_exp_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_exp_dims)


global graph
graph = tf.get_default_graph()
print("Loading AEYE model")
get_model()
print(labels)
# path = 'C:\\Users\\levyc1\\AEYE'


@app.route("/predict", methods=["POST"])
def predict():
    print("API predict called")
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    print("prepare_image")
    preprocessed_image = prepare_image(image)
    print("predict")
    with graph.as_default():
        prediction = model.predict(preprocessed_image)
    pred_class = prediction.argmax(axis=-1)
    pred_perc = max(prediction[0])
    label = labels[pred_class[0]]
    print(label)
    response = {
        'label': label,
        'percentage': pred_perc
    }
    return jsonify(response)


@app.route('/test')
def running():
    return 'App is running'


@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)


if __name__ == '__main__':
   app.run(host='127.0.0.1', debug=True, port=9050)

# c = CaptionBot()
# print(c.url_caption('https://en.wikipedia.org/wiki/Lenna#/media/File:Lenna_(test_image).png'))
# c.file_caption('your local image filename here')
