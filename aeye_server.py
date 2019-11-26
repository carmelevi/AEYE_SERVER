import numpy as np
import keras
import base64
import io
import json
import requests
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from flask import request
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask import Response
from captionbot import CaptionBot
from aeye_labels import labels

app = Flask(__name__)
CORS(app)

def get_model():
    global model
    model = load_model('aeye_model_a.h5')
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
    # prediction = model.predict(preprocessed_image)
    pred_class = prediction.argmax(axis=-1)
    pred_perc = max(prediction[0])
    label_en = labels['EN'][pred_class[0]]
    label_it = labels['IT'][pred_class[0]]
    print(label_en)
    response = {
        'label_en': label_en,
        'label_it': label_it,
        'percentage': str(pred_perc)
    }
    return Response(json.dumps(response), 200, {'Access-Control-Allow-Origin': '*'})


@app.route("/caption_bot", methods=["POST"])
def caption_bot():
    print("API caption bot called")
    message = request.get_json(force=True)
    encoded = message['image']
    print(encoded)
    url = 'https://api.imgbb.com/1/upload'
    data = {
        'key': '4bbdc1539b907002d0b9654c060ec60c',
        'image': encoded
    }
    upload_img = requests.post(url, data=data)
    response_dict = json.loads(upload_img.text)
    c = CaptionBot()
    caption = c.url_caption(response_dict['data']['url'])
    print(caption)
    response = {
        'caption': caption
    }
    return Response(json.dumps(response), 200, {'Access-Control-Allow-Origin': '*'})


@app.route('/ping', methods=['GET'])
def hello():
    response = {
        'greeting': 'Pong'
    }
    return Response(json.dumps(response), 200, {'Access-Control-Allow-Origin': '*'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=9050)

