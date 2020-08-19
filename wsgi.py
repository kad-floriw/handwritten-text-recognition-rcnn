import os
import cv2
import math
import flask
import numpy as np
import tensorflow as tf

from src.DataLoader import Batch
from flask import request, jsonify
from src.Model import Model, DecoderType
from src.SamplePreprocessor import preprocess

model, graph = None, None
vocabulary = '.0123456789'
app = flask.Flask(__name__)


@app.route('/recognize', methods=['POST'])
def recognize():
    file_keys = list(request.files.keys())
    batch_count = math.ceil(len(file_keys) / model.batchSize)

    if batch_count:
        detections = []

        for i in range(batch_count):
            start = i * model.batchSize
            stop = start + model.batchSize

            images = [
                cv2.imdecode(np.frombuffer(request.files[key].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                for key in file_keys[start:stop]
            ]

            batch = Batch(None, list(map(lambda x: preprocess(x, Model.imgSize), images)))
            with graph.as_default():
                b_recognized, b_probabilities = model.inferBatch(batch, True)
            b_probabilities = [0] * len(b_recognized) if b_probabilities is None else b_probabilities.tolist()

            detections += [{
                'detected': recognized,
                'probability': probability
            } for recognized, probability in zip(b_recognized, b_probabilities)]

        result = {
            'success': True,
            'result': detections
        }
    else:
        result = {
            'success': False
        }

    return jsonify(result)


@app.before_first_request
def before_first_request():
    global model, graph

    graph = tf.Graph()
    weight_location = os.environ.get('WEIGHTS', os.path.join('weights'))
    with graph.as_default():
        model = Model(vocabulary, graph, DecoderType.BestPath, model_dir=weight_location)


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, use_reloader=False)
